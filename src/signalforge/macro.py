"""Registered date-bounded FRED snapshot/revision requests; credentials from env."""
import os
import json
from pathlib import Path
import pandas as pd
from .sources import INDEX,parse_fred
from .runtime import digest,file_hash

SERIES={'CPIAUCSL':'M','INDPRO':'M','UNRATE':'M','DGS2':'D','DGS10':'D'}


def metadata_request(series,vintage):
    if series not in SERIES or pd.Timestamp(vintage)>=pd.Timestamp('2024-01-01'):raise PermissionError('Registered pre-reserved metadata vintage required')
    date=pd.Timestamp(vintage).date().isoformat()
    return {'url':'https://api.stlouisfed.org/fred/series','params':{'series_id':series,'realtime_start':date,'realtime_end':date,'file_type':'json'}}


def canonical_macro_events(series,raw_events,unit_evidence):
    """Unit evidence must match the observation vintage, never today's index base."""
    from .sources import allowed_url
    events=[];blocked=[]
    for value in raw_events:
        vintage=value.get('realtime_start')
        evidence=unit_evidence.get(vintage,{})
        if not vintage or not evidence:
            blocked.append({'series':series,'vintage':vintage,'reference_time':value['reference_time'],'state':'BLOCKED_UNIT_VINTAGE'});continue
        request=metadata_request(series,vintage)
        if evidence.get('source_url')!=request['url'] or not allowed_url(evidence['source_url']) or file_hash(evidence['path'])!=evidence['sha256']:
            raise ValueError('Macro unit metadata evidence invalid')
        body=json.loads(Path(evidence['path']).read_text());rows=body.get('seriess',[])
        if len(rows)!=1 or rows[0].get('id')!=series or rows[0].get('frequency_short')!=SERIES[series]:raise ValueError('Macro series/frequency metadata mismatch')
        metadata=rows[0]
        if metadata.get('realtime_start')!=vintage or metadata.get('realtime_end')!=vintage or not metadata.get('units'):
            raise ValueError('Macro metadata must be explicitly pinned to observation vintage')
        known=pd.Timestamp(evidence['public_available_at'])
        if known.tzinfo is None or known>pd.Timestamp(value['available_at']):raise ValueError('Macro unit evidence not known by feature availability')
        events.append({**value,'entity':series,'source':'macro','field':'value','unit':metadata['units'],
            'unit_metadata_hash':evidence['sha256'],'unit_metadata_vintage':vintage,'original_publication_qualified':False})
    if events:
        from .data import validate_events
        validate_events(pd.DataFrame(events)) # Rebases cannot silently mix units.
    return events,blocked


def request_plan(series,start='2010-07-20',end='2023-12-31'):
    if series not in SERIES or pd.Timestamp(start)>pd.Timestamp(end) or pd.Timestamp(end)>=pd.Timestamp('2024-01-01'):
        raise PermissionError('Registered macro series and pre-reserved bounded dates required')
    base={'series_id':series,'observation_start':start,'observation_end':end,'file_type':'json','limit':10000,'sort_order':'asc','units':'lin'}
    return [{'url':INDEX['fred'],'params':{**base,'realtime_start':start,'realtime_end':start,'output_type':1},'stage':'initial_snapshot'},
            {'url':INDEX['fred'],'params':{**base,'realtime_start':str((pd.Timestamp(start)+pd.Timedelta(days=1)).date()),'realtime_end':end,'output_type':3},'stage':'new_and_revised'}]


def acquire_series(client,series,start='2010-07-20',end='2023-12-31',maximum_pages=20,unit_evidence=None):
    plan=request_plan(series,start,end)
    key=os.environ.get('FRED_API_KEY')
    if not key:raise PermissionError('BLOCKED_AUTH: FRED_API_KEY absent')
    receipts=[];events=[]
    for request in plan:
        offset=0
        for page in range(maximum_pages):
            params={**request['params'],'api_key':key,'offset':offset}
            raw=client.cached(request['url'],'json',params) or client.fetch(request['url'],'json',params,
                {'stage':request['stage'],'series':series,'request_plan_hash':digest(plan)})
            body=json.loads(Path(raw['path']).read_text())
            if not {'observations','count','offset','limit'}<=set(body) or int(body['offset'])!=offset:
                raise ValueError('Macro pagination schema/offset mismatch')
            values=parse_fred(raw['path'],SERIES[series]);receipts.append(raw)
            for value in values:
                if pd.Timestamp(value['reference_label'])>pd.Timestamp(end) or pd.Timestamp(value['available_at'])>pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=2):
                    raise PermissionError('Provider exceeded bounded observation/vintage request')
                events.append({**value,'entity':series,'source':'fred','field':'value','raw_hash':raw['sha256']})
            offset+=len(body['observations'])
            if offset>=int(body['count']):break
            if not body['observations']:raise ValueError('Empty incomplete pagination')
        else:raise RuntimeError('PAUSED_BUDGET: macro pagination ceiling reached')
    canonical,blocked=canonical_macro_events(series,events,unit_evidence or {})
    return {'series':series,'state':'SUCCEEDED_RAW','events':events,'canonical_events':canonical,'unit_vintage_blockers':blocked,
        'canonical_state':'BLOCKED_UNIT_VINTAGE' if blocked else 'SUCCEEDED_RECONSTRUCTED_CANONICAL',
        'receipts':receipts,'pit_tier':'B','intraday_original_publication_qualified':False}
