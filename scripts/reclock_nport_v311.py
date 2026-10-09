"""Re-clock existing N-PORT canonical events with a delayed Tier-B availability assumption."""
import json
from pathlib import Path
import pandas as pd
from signalforge.runtime import paths,atomic_json,commit_bundle,digest,file_hash,now
from signalforge.data import validate_events
from signalforge.nport_reclock import conservative_development_clock

repo,runtime=paths()
source_path=repo/'reports/nport_bulk_integration.json'
if not source_path.exists():raise RuntimeError('BLOCKED_DATA: parent N-PORT integration absent')
parent=json.loads(source_path.read_text())
if parent.get('state')!='SUCCEEDED_RECONSTRUCTED_NPORT_TARGET_FLOWS':
    raise RuntimeError('BLOCKED_DATA: parent N-PORT integration not successful')
folder=(runtime/parent['events_relative_path']).parent
events=json.loads((folder/'events.json').read_text())
clocks=json.loads((folder/'clocks.json').read_text())
accepted={row['ACCESSION_NUMBER']:pd.Timestamp(row['accepted_at']) for row in clocks}
out=[]
for row in events:
    accession=row.get('accession')
    if accession not in accepted:raise ValueError('N-PORT event lacks Accepted-time evidence')
    event=dict(row)
    event['available_at']=conservative_development_clock(accepted[accession]).isoformat()
    event['clock_policy']='edgar_next_federal_business_day_235959_et_development_v311'
    event['clock_evidence_class']='accepted_timestamp_plus_conservative_unverified_development_delay'
    event['original_publication_qualified']=False
    out.append(event)
frame=pd.DataFrame(out)
validate_events(frame)
keys=['entity','source','field','reference_time','available_at']
if frame.duplicated(keys).any():raise ValueError('Re-clocked N-PORT events duplicate canonical key')
payload=json.loads(frame.sort_values(['available_at','entity','field','reference_time']).to_json(orient='records'))
identity=digest({'parent_artifact_id':parent['artifact_id'],'clock_policy':'next_federal_business_day_235959_et_v311','events':payload})
dest=runtime/'artifacts/nport_canonical_v311'/identity
commit_bundle(dest,{'events.json':payload},{
    'parent_artifact_id':parent['artifact_id'],'pit_tier':'B','qualified_for_final':False,
    'clock_policy':'next_federal_business_day_235959_et_v311','reserved_access':False
})
result=dict(parent)
result.update({
    'state':'SUCCEEDED_RECONSTRUCTED_NPORT_TARGET_FLOWS',
    'artifact_id':identity,
    'parent_artifact_id':parent['artifact_id'],
    'events_relative_path':str((dest/'events.json').relative_to(runtime)),
    'events_sha256':file_hash(dest/'events.json'),
    'created_at':now(),
    'clock_policy':'edgar_next_federal_business_day_235959_et_development_v311',
    'parent_clock_policy':'edgar_accepted_plus_15m_parent_protocol',
    'original_publication_qualified':False,'qualified_for_final':False,'pit_tier':'B',
    'claim_boundary':'Same parent Form N-PORT values and Accepted-time evidence, assigned to 23:59:59 ET on the next US-federal business day for conservative Tier-B development. SEC provides no guaranteed first-public-availability timestamp, so this is not strict PIT or final-publication proof.'
})
# Keep large parent quarter receipts out of the successor summary; parent receipt remains the immutable source.
result.pop('quarter_receipts',None)
atomic_json(repo/'reports/nport_bulk_integration_v311.json',result)
print(json.dumps(result,indent=2))
