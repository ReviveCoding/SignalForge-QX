"""Resume explicit keyed source plans without rerunning other acquisitions."""
import json,os
from signalforge.runtime import paths,atomic_json,now
from signalforge.source_requests import keyed_pilot
repo,runtime=paths();records=[]
from signalforge.source_requests import validate_fred_plan,fred_plan_path
from signalforge.runtime import file_hash
from signalforge.tiingo import register_plan,validate_plan
fred_path=fred_plan_path(repo);fred_hash=file_hash(fred_path) if fred_path.is_file() else None
tiingo_plan=register_plan(repo,runtime)
for source in ['tiingo','tiingo_actions','fred','nport','prices']:
    try:records.append(keyed_pilot(repo,runtime,source))
    except (PermissionError,RuntimeError,ValueError,KeyError,TypeError) as error:
        message=str(error)
        state='BLOCKED_AUTH' if message.startswith('BLOCKED_AUTH:') else 'BLOCKED_REQUEST_OR_SOURCE'
        if source=='tiingo_actions' and message.startswith('BLOCKED_AUTH: HTTP 403'):
            # The same token already qualified the bounded EOD branch; do not retry/bypass
            # a separately inaccessible corporate-actions endpoint.
            state='BLOCKED_ENTITLEMENT'
        record={'source':source,'state':state,'reason':message,'reserved_access':False,'model_qualified':False}
        if source=='nport':record['dissemination']={'state':'BLOCKED_DATA','reason':'Independent historical dissemination evidence still required'}
        if source=='prices':record.update(qualification='PRELIMINARY_LEGACY',source_time_scope_state='BLOCKED_SOURCE_TIME_SCOPE')
        records.append(record)
result={'state':'PARTIAL' if any(r['state'].startswith('SUCCEEDED') for r in records) else 'BLOCKED_DATA',
    'sources':records,'created_at':now(),'reserved_access':False,'scientific_qualified':False,
    'credential_presence':{k:bool(os.environ.get(k)) for k in ['FRED_API_KEY','TIINGO_API_TOKEN','SEC_USER_AGENT','ALPHAVANTAGE_API_KEY']},
    'fred_plan_sha256':fred_hash,'tiingo_plan_id':validate_plan(tiingo_plan)}
if fred_hash is not None and file_hash(fred_path)!=fred_hash:raise ValueError('Existing FRED plan changed')
atomic_json(repo/'reports/keyed_source_access.json',result);print(json.dumps(result,indent=2))
