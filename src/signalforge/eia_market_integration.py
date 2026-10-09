"""Map corrected EIA petroleum development evidence only to economically relevant USO features."""
import json
from pathlib import Path
import pandas as pd
from .data import eia_events,validate_events
from .runtime import digest,commit_bundle,atomic_json,file_hash,now

FIELD_MAP={
 'Commercial (Excluding SPR)_value':'commercial_crude_value',
 'Commercial (Excluding SPR)_change':'commercial_crude_change',
 'Cushing_value':'cushing_value','Cushing_change':'cushing_change',
 'Total Motor Gasoline_change':'gasoline_change','Distillate Fuel Oil_change':'distillate_change'}


def build_canonical(repo,runtime):
    repo,runtime=Path(repo),Path(runtime);path=repo/'reports/eia_development_acquisition.json'
    audit=repo/'reports/eia_corrected_source_audit.json';incident=repo/'reports/source_integrity_incident.json'
    if not path.exists() or not audit.exists() or not incident.exists():raise RuntimeError('BLOCKED_DATA: corrected EIA evidence set incomplete')
    q=json.loads(audit.read_text());i=json.loads(incident.read_text());records=json.loads(path.read_text())['records']
    if q.get('state')!='PASSED_CORRECTED_DEVELOPMENT_SOURCE' or i.get('state')!='CORRECTED_ALL_FAMILY_DEVELOPMENT_RECOMPUTED_RELOAD_VERIFIED':
        raise PermissionError('Corrected EIA source incident not qualified for development reuse')
    events=eia_events(records)
    events=events[events.field.isin(FIELD_MAP)].copy();events['field']=events.field.map(FIELD_MAP);events['entity']='USO'
    events['original_publication_qualified']=False;events['economic_applicability']='USO petroleum exposure proxy only'
    validate_events(events)
    if events.duplicated(['entity','source','field','reference_time','available_at']).any():raise ValueError('Duplicate EIA canonical event')
    payload=json.loads(events.sort_values(['available_at','field']).to_json(orient='records'))
    identity=digest({'acquisition_sha':file_hash(path),'audit_sha':file_hash(audit),'incident_sha':file_hash(incident),'events':payload})
    folder=runtime/'artifacts/eia_market_canonical'/identity
    commit_bundle(folder,{'events.json':payload},{'pit_tier':'B','asset':'USO','qualified_for_final':False})
    result={'state':'SUCCEEDED_RECONSTRUCTED_EIA_USO_ACTIVITY','artifact_id':identity,
        'events_relative_path':str((folder/'events.json').relative_to(runtime)),'events_sha256':file_hash(folder/'events.json'),
        'rows':len(events),'fields':sorted(events.field.unique()),'assets':['USO'],'pit_tier':'B','qualified_for_final':False,
        'original_publication_qualified':False,'reserved_access':False,'whole_archive_complete':bool(q.get('whole_archive_complete',False)),
        'calibration_source_consistency':'BLOCKED_SOURCE_CONSISTENCY','created_at':now(),
        'claim_boundary':'Weekly petroleum physical-activity proxy for USO only; not applicable to other assets, not causal oil demand, and no zero fill for non-applicable sources.'}
    atomic_json(repo/'reports/eia_market_integration.json',result);return result
