"""Read only immutable registered development FRED pages; never inspect credentials."""
import json
from pathlib import Path
from signalforge.runtime import paths, file_hash, atomic_json, now, commit_bundle, digest
from signalforge.source_requests import decode_fred_page, validate_fred_plan

repo,runtime=paths()
plan=json.loads((repo/'.local/fred_request_plan_v31.json').read_text(encoding='utf-8-sig'))
plan_id=validate_fred_plan(plan,require_json_safe_segments=True)
pages=[]
for receipt in sorted((runtime/'raw').glob('*/*.receipt.json')):
    raw=json.loads(receipt.read_text())
    if raw.get('metadata',{}).get('series_id') not in {s['series_id'] for s in plan['series']}:
        continue
    path=Path(raw['path'])
    if file_hash(path)!=raw['sha256']:raise ValueError('Immutable FRED cache corruption')
    body=json.loads(path.read_text())
    matches=[s for s in plan['series'] if s['series_id']==raw['metadata']['series_id']
             and s['output_type']==body.get('output_type')
             and s['realtime_start']==body.get('realtime_start') and s['realtime_end']==body.get('realtime_end')]
    if not matches:continue # Parent responses remain preserved, outside successor registration.
    rows=body['observations']
    entry={'series_id':matches[0]['series_id'],'output_type':body['output_type'],
           'realtime_start':body['realtime_start'],'realtime_end':body['realtime_end'],
           'count':body['count'],'offset':body['offset'],'limit':body['limit'],
           'observation_rows':len(rows),'vintage_cells':sum(len(x)-1 for x in rows) if body['output_type']==3 else len(rows),
           'raw_sha256':raw['sha256'],'receipt_path':str(receipt)}
    try:
        values,count,progress=decode_fred_page(raw,matches[0],body['offset'],plan['page_size'])
        entry.update(decoder_state='ACCEPTED',observation_versions=len(values),pagination_progress=progress)
    except (ValueError,PermissionError) as error:
        entry.update(decoder_state='REJECTED',error_type=type(error).__name__,reason=str(error))
    pages.append(entry)
    if matches[0]['series_id']=='DGS10' and body['output_type']==3 and body['count']==4002:
        dest=repo/'tests/fixtures/fred'/f'dgs10_revision_offset_{body["offset"]}.json'
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and file_hash(dest)!=raw['sha256']:raise ValueError('Regression fixture bytes changed')
        if not dest.exists():dest.write_bytes(path.read_bytes())
result={'state':'AUDITED_IMMUTABLE_REGISTERED_FRED_PAGES','created_at':now(),'plan_id':plan_id,
        'pages':pages,'reserved_access':False,'acquisition_complete':False,
        'claim_boundary':'Cached page audit only; missing offset windows and request segments are not completed acquisition.'}
prior=repo/'reports/fred_cached_page_audit.json'
if prior.exists():
    old=json.loads(prior.read_text())
    commit_bundle(runtime/'artifacts/fred_cached_page_audits'/digest(old),{'audit.json':old},{'reserved_access':False})
commit_bundle(runtime/'artifacts/fred_cached_page_audits'/digest(result),{'audit.json':result},{'reserved_access':False})
atomic_json(prior,result)
print(json.dumps(result,indent=2))
