"""Reuse the discovered official index; bounded 2010–2023 annual archive audit."""
import json
from pathlib import Path
import time
import psutil
import shutil
import pandas as pd
from signalforge.runtime import paths,atomic_json,file_hash,now
from signalforge.sources import Acquisition,INDEX,discover_zip,parse_cftc
from signalforge.runtime import digest
import inspect


def main():
    repo,runtime=paths();started=time.monotonic();manifest=repo/'reports/cftc_development_acquisition.json'
    host=json.loads((repo/'reports/host_disk_evidence.json').read_text())
    if (pd.Timestamp.now(tz='UTC')-pd.Timestamp(host['observed_at'])>pd.Timedelta(hours=2) or
        host['free_bytes']<21*1024**3 or shutil.disk_usage(runtime).free<11*1024**3 or psutil.virtual_memory().available<2*1024**3):
        raise RuntimeError('BLOCKED_RESOURCE: CFTC host/runtime/RAM admission')
    previous=json.loads(manifest.read_text()) if manifest.exists() else {'records':[]}
    records={(r['year'],r['family']):r for r in previous['records']}
    parser_id=digest(inspect.getsource(parse_cftc))
    docs=json.loads((repo/'reports/source_document_verification.json').read_text())
    index=next(row['raw'] for row in docs['sources'] if row['source']=='cftc' and 'raw' in row)
    if file_hash(index['path'])!=index['sha256']:raise ValueError('Discovered CFTC index snapshot corrupt')
    html=Path(index['path']).read_text(errors='replace');client=Acquisition(runtime,128*1024**2)
    plan={'created_at':now(),'years':list(range(2010,2024)),'families':['tff','disaggregated'],'files':28,
          'source_index_sha256':index['sha256'],'byte_ceiling':128*1024**2,'segment_seconds':1200,'workers':1,
          'pilot_bytes_per_2022_pair':2543752,'estimated_bytes_from_2022_pair':2543752*14,
          'host_evidence':host,'ram_estimate_bytes':1024**3,'reserved_access':False}
    atomic_json(repo/'reports/cftc_acquisition_plan.json',plan)
    for year in plan['years']:
        for family in plan['families']:
            prior=records.get((year,family))
            if prior:
                if prior['state']=='SUCCEEDED':
                    if file_hash(prior['raw']['path'])!=prior['raw']['sha256']:raise ValueError('Corrupt committed annual archive')
                    continue
                corrected_schema=prior['state']=='FAILED_SOURCE' and prior.get('parser_id')!=parser_id
                if not corrected_schema and (prior['state']!='FAILED_RETRYABLE' or prior.get('attempts',1)>=2):continue
            before=time.monotonic()
            try:
                url=discover_zip(html,'cftc',year,family)
                raw=client.cached(url,'zip')
                if prior and prior['state']=='FAILED_SOURCE' and raw is None:raise ValueError('Schema recovery requires cached bytes; no source retry')
                raw=raw or client.fetch(url,'zip',metadata={'year':year,'family':family,'scope':'pre2024_development'})
                frame=parse_cftc(raw['path'],family)
                if frame.report_date.dt.year.ne(year).any():raise ValueError('Archive year schema mismatch')
                audit={'rows':len(frame),'contracts':int(frame.CFTC_Contract_Market_Code.nunique()),'report_dates':int(frame.report_date.nunique()),
                       'first_report':str(frame.report_date.min()),'last_report':str(frame.report_date.max()),
                       'open_interest_missing':int(frame.Open_Interest_All.isna().sum()),'open_interest_zero':int(frame.Open_Interest_All.eq(0).sum()),
                       'parsed_memory_bytes':int(frame.memory_usage(deep=True).sum())}
                record={'year':year,'family':family,'state':'SUCCEEDED','raw':raw,'audit':audit,'pit_tier':'B_UNQUALIFIED_ORIGINAL_VINTAGE_RELEASE',
                        'available_at':None,'model_qualified':False,'positions_are_cashflows':False,'seconds':time.monotonic()-before}
                del frame
            except Exception as exc:
                retryable=isinstance(exc,RuntimeError) and ('Network failure' in str(exc) or str(exc).startswith('HTTP 5'))
                record={'year':year,'family':family,'state':'BLOCKED_AUTH' if isinstance(exc,PermissionError) else 'FAILED_RETRYABLE' if retryable else 'FAILED_SOURCE',
                        'reason':str(exc),'seconds':time.monotonic()-before}
            record['attempts']=(prior or {}).get('attempts',0)+1;records[(year,family)]=record
            record['parser_id']=parser_id
            if prior:record['previous_attempt']=prior
            successful=sum(r['state']=='SUCCEEDED' for r in records.values())
            atomic_json(manifest,{'plan':plan,'records':[records[k] for k in sorted(records)],'completed':successful,'required':28,
                                 'state':'SUCCEEDED_RAW' if successful==28 else 'PARTIAL','elapsed_seconds':time.monotonic()-started,
                                 'historical_PIT_qualified':False,'reserved_access':False})
            print(json.dumps({'source':'cftc','year':year,'family':family,'state':record['state'],'completed':successful,'required':28}),flush=True)
            if record['state']=='BLOCKED_AUTH' or time.monotonic()-started>=1200:return


if __name__=='__main__':main()
