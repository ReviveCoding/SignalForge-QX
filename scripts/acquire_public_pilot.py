"""Real bounded source discovery/acquisition; independent branches retain failures."""
import json
from pathlib import Path
from signalforge.runtime import paths, atomic_json, now
from signalforge.sources import Acquisition, INDEX, discover_zip, discover_eia, parse_cftc

repo, runtime = paths()
client = Acquisition(runtime, 128*1024**2)
results = []
for source in ['cftc','eia','nport','fred','prices']:
    try:
        if source == 'cftc':
            html, index = client.html(INDEX[source])
            for family in ['tff','disaggregated']:
                receipt = client.fetch(discover_zip(html, source, 2022, family), 'zip', metadata={'family':family,'year':2022})
                frame = parse_cftc(receipt['path'], family)
                results.append({'source':source, 'family':family,'state':'SUCCEEDED_RAW', 'raw':receipt,
                                'audit':{'rows':len(frame),'entities':int(frame.CFTC_Contract_Market_Code.nunique()),
                                         'dates':int(frame.report_date.nunique()), 'columns':list(frame),
                                         'min_date':str(frame.report_date.min()),'max_date':str(frame.report_date.max()),
                                         'open_interest_zero':int(frame.Open_Interest_All.eq(0).sum()),
                                         'open_interest_missing':int(frame.Open_Interest_All.isna().sum())},
                                'pit_tier':'B_UNQUALIFIED_TIMING', 'model_qualified':False})
        elif source == 'eia':
            page = 'https://www.eia.gov/petroleum/supply/weekly/archive/2022/2022_01_12/wpsr_2022_01_12.php'
            html, index = client.html(page)
            for table in ['1','4','9']:
                receipt = client.fetch(discover_eia(html,page,table), 'csv', metadata={'release_page':page,'table':table})
                results.append({'source':source,'state':'SUCCEEDED_RAW','table':table,'raw':receipt,'model_qualified':False})
        elif source == 'nport':
            import os
            if not os.environ.get('SEC_USER_AGENT'):
                raise PermissionError('BLOCKED_AUTH: genuine SEC_USER_AGENT absent')
            html,index=client.html(INDEX[source])
            receipt=client.fetch(discover_zip(html,source,'2022Q1'),'zip')
            results.append({'source':source,'state':'SUCCEEDED_RAW','raw':receipt,'model_qualified':False})
        else:
            from signalforge.source_requests import keyed_pilot
            results.append(keyed_pilot(repo,runtime,source))
    except Exception as e:
        results.append({'source':source,'state':'BLOCKED_AUTH' if isinstance(e,PermissionError) else 'FAILED_SOURCE',
                        'reason':str(e),'error_type':type(e).__name__})
result={'created_at':now(),'byte_budget':128*1024**2,'remaining_bytes':client.remaining,'sources':results,
        'research_results':False,'reserved_data_accessed':False}
atomic_json(repo/'reports/public_pilot.json',result)
print(json.dumps(result,indent=2))
