"""Register the preregistered Main/Nested source-to-target mapping as immutable evidence."""
import json
from signalforge.runtime import paths, atomic_json, file_hash, digest, now
repo,runtime=paths()
path=repo/'configs/track_source_mapping_v31.json'
cfg=json.loads(path.read_text())
if cfg.get('outcome_blind') is not True:raise PermissionError('Outcome-blind source mapping required')
if cfg.get('rules',{}).get('cftc_semantics')!='reported_positioning_contracts_not_cash_flow':
    raise ValueError('CFTC positioning semantics changed')
expected={'SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'}
if set(cfg.get('targets',{}))!=expected:raise ValueError('Frozen eight-asset mapping required')
if cfg['rules'].get('unmatched_source')!='missing_not_zero':raise ValueError('Unmatched source must remain missing')
receipt={'state':'PREREGISTERED_TRACK_SOURCE_MAPPING','registered_at':now(),'reserved_access':False,
    'qualified_source_clocks':False,'config_path':'configs/track_source_mapping_v31.json',
    'config_sha256':file_hash(path),'targets':sorted(expected),'outcome_blind':True,
    'claim_boundary':'semantic preregistration only; not source-clock/entity/final qualification'}
receipt['artifact_id']=digest(receipt)
atomic_json(repo/'reports/track_source_mapping_v31.json',receipt)
print(json.dumps(receipt,indent=2))
