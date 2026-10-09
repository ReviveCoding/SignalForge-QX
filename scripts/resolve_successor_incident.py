"""Close a preserved GPU incident only after bounded verified implementation repair."""
import argparse,json
from signalforge.runtime import paths,atomic_json,now,code_hash,digest,commit_bundle
from signalforge.completion import current_full_validation
from signalforge.successor import successor_authorized,load_frozen_plan,assert_parent_unchanged

p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);args=p.parse_args()
repo,runtime=paths();path=repo/'reports/successor_gpu_incident.json';incident=json.loads(path.read_text())
verdict=json.loads((repo/'reports/auto_recovery_verdict.json').read_text(encoding='utf-8-sig'))
if args.attempt not in (1,2):raise PermissionError('Repair attempt ceiling exceeded')
if verdict.get('classification') not in {'CODE_ORCHESTRATION_DEFECT','SAFE_TRANSIENT_RETRY','CUDA_RUNTIME_OR_LIBRARY_DEFECT'}:
    raise PermissionError('Unsafe gate cannot be repaired automatically')
if verdict.get('state') not in {'FIX_VERIFIED','SAFE_RETRY'} or not verdict.get('safe_to_resume_gpu') or not verdict.get('successor_authorization_verified'):
    raise PermissionError('Verified GPU recovery verdict absent')
if verdict.get('scientific_contract_changed') is not False or verdict.get('budget_or_ledger_changed') is not False:
    raise PermissionError('Recovery cannot change scientific contracts or ledgers')
if not successor_authorized(repo) or not current_full_validation(repo)['passed']:raise PermissionError('Current authorized full validation required')
load_frozen_plan(repo,runtime)
assert_parent_unchanged(runtime,incident['ledgers']['development_compute.sqlite'])
resolution={'incident_id':incident['incident_id'],'verdict':verdict,'repair_attempt':args.attempt,'resolved_at':now(),
            'source_tree_hash':code_hash(repo),'reserved_access':False}
directory=runtime/'artifacts/successor_gpu_repairs'/digest(resolution)
commit_bundle(directory,{'resolution.json':resolution},{'incident_id':incident['incident_id'],'repair_attempt':args.attempt})
atomic_json(path,{**incident,'state':'RESOLVED_VERIFIED','repair_attempt':args.attempt,'resolution_relative_bundle':str(directory.relative_to(runtime))})
