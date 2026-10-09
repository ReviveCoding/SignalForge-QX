"""Measured track-specific v3.1 successor CUDA pilot. Cost/compatibility only; no model selection."""
import argparse,json,sqlite3
from signalforge.runtime import paths,atomic_json,gpu_lease,now,file_hash,validate_bundle,digest,commit_bundle,code_hash
from signalforge.completion import load_protocol,successor_track_boundary,current_full_validation
from signalforge.track_inputs import prepare_track_inputs
from signalforge.track_engine import validate_panel,prepare,_fit_registered,track_operator_id
from signalforge.panels import track_folds
from signalforge.experiments import ComputeBudget
from signalforge.successor import require_successor_access,validate_pilot,ledger_snapshot,load_frozen_plan,assert_parent_unchanged,verified_pilot_retry
from signalforge.successor_inputs import prepared_track
from signalforge.successor_runtime import progress,preserve_incident
from signalforge.successor_backend import activate_backend,operator_id

p=argparse.ArgumentParser();p.add_argument('--track',choices=['Main-A','Nested-B'],required=True);a=p.parse_args()
repo,runtime=paths();track=a.track
require_successor_access(repo)
plan,plan_pointer=load_frozen_plan(repo,runtime)
parent=ledger_snapshot(runtime/'ledger/development_compute.sqlite',43200)
if not current_full_validation(repo)['passed']:
    raise RuntimeError('BLOCKED_VALIDATION: current full source-bound validation and exact qualification required')
safe=track.replace('-','_')
receipt_path=repo/'reports'/('successor_gpu_pilot_'+safe+'.json')
retry_attempt=0
partial_path=repo/'reports'/('successor_gpu_pilot_progress_'+safe+'.json')
prior_results=json.loads(partial_path.read_text()).get('results',[]) if partial_path.exists() else []
if receipt_path.exists():
    existing=json.loads(receipt_path.read_text())
    if existing.get('state')=='SUCCEEDED_MEASURED_SUCCESSOR_PILOT':
        validate_pilot(repo,runtime,track)
        print(json.dumps({k:v for k,v in existing.items() if k not in {'results','charges'}},indent=2))
        raise SystemExit(0)
    prior_results=existing.get('results',prior_results)
    commit_bundle(runtime/'artifacts/successor_pilot_attempts'/digest(existing),{'pilot.json':existing},{'track':track,'reserved_access':False})
if prior_results or receipt_path.exists():
    incident_path=repo/'reports/successor_gpu_incident.json'
    incident=json.loads(incident_path.read_text()) if incident_path.exists() else {}
    retry_attempt=verified_pilot_retry(receipt_path.exists(),prior_results,incident)

protocol=load_protocol(repo)
boundary=successor_track_boundary(repo,track)
if boundary.get('state')!='READY_FOR_MEASURED_SUCCESSOR_PILOT':
    raise RuntimeError('BLOCKED_DATA: '+boundary.get('reason','successor pilot boundary not ready'))

pilot=protocol['successor_gpu'];families=pilot['families']
if families!=['lightgbm','xgboost','mlp','gru','rgmf_linear','rgmf_gru']:
    raise PermissionError('Successor pilot family preregistration changed')
fixed={'lightgbm':1.0,'xgboost':1.0,'mlp':[16,0.001],'gru':[16,0.001],
       'rgmf_linear':[16,0.001],'rgmf_gru':[16,0.001]}
seed=pilot['seeds'][0];planned=len(families)
if 2*planned>pilot['pilot_fit_ceiling']:raise PermissionError('Successor total pilot plan exceeds fit ceiling')

study=json.loads((repo/'configs/study.json').read_text())
info='I3' if track=='Main-A' else 'I4'
manifest_path=repo/'.local'/('inputs_'+track+'_v311.json')
manifest=json.loads(manifest_path.read_text())
progress(repo,runtime,'SUCCESSOR GPU PILOT '+('MAIN' if track=='Main-A' else 'NESTED'),track=track,information_set=info,year=2022,seed=seed,completed_fits=0,planned_fits=planned,status='PREPARING_HASH_BOUND_PANELS')
panels,contexts,q=prepared_track(repo,runtime,track)
if q.get('blocked_information_sets') or q.get('state')!='READY_RECONSTRUCTED_DEVELOPMENT_INPUTS':
    raise RuntimeError('BLOCKED_DATA: complete registered information grid required for '+track)
panel=panels[info];x=validate_panel(panel,contexts[info]);folds,fail=track_folds(panel,study,track)
if fail:raise RuntimeError('BLOCKED_DATA: pilot track fold failure for '+track)
year=2022;fold=folds[year]
training=fold['train'].loc[fold['train'].context_eligible].copy()
training.attrs.update(fold['train'].attrs)
prepared=prepare(training,fold['test'],x,fold['outer_cutoff'])
backend=activate_backend(repo,runtime);grid_id=q['grid_id']
successful_prior={r['family']:r for r in prior_results if r.get('state')=='SUCCEEDED'}
for family,row in successful_prior.items():
    operator=operator_id(repo,study,track,family,backend)
    if family not in families or row.get('regularization')!=fixed[family] or row.get('seed')!=seed:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: prior pilot recipe differs')
    bundle=runtime/'artifacts/track_development'/row['run_id'];receipt=validate_bundle(bundle)
    if receipt['metadata']['training_operator_id']!=operator or json.loads((bundle/'metrics.json').read_text())!=row:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: prior successful pilot mathematics changed')
    validate_bundle(runtime/row['model_relative_bundle'])

budget_path=runtime/'ledger/successor_gpu_pilot_compute.sqlite'
budget=ComputeBudget(budget_path,pilot['pilot_gpu_seconds']);results=[]
try:
    with gpu_lease(runtime):
        for family in families:
            operator=operator_id(repo,study,track,family,backend)
            require_successor_access(repo)
            if family in successful_prior:
                results.append(successful_prior[family]);continue
            bill=ledger_snapshot(budget_path,pilot['pilot_gpu_seconds'])
            if bill['entries']>=pilot['pilot_fit_ceiling']:raise RuntimeError('RESOURCE_OR_BUDGET_GATE: pilot cumulative fit ceiling reached')
            progress(repo,runtime,'SUCCESSOR GPU PILOT '+('MAIN' if track=='Main-A' else 'NESTED'),track=track,family=family,information_set=info,year=year,seed=seed,completed_fits=len(results),planned_fits=planned,status='FITTING')
            results.append(_fit_registered(
                repo,runtime,prepared,fold['test'],family,fixed[family],seed,
                track,info,year,'successor_pilot_'+safe+('_repair'+str(retry_attempt) if retry_attempt else ''),operator,grid_id,budget,
                'successor_gpu_cost_pilot_v311'))
            atomic_json(repo/'reports'/('successor_gpu_pilot_progress_'+safe+'.json'),{'track':track,'results':results,'created_at':now(),'source_tree_hash':code_hash(repo),'reserved_access':False})
            assert_parent_unchanged(runtime,parent)
except Exception as error:
    preserve_incident(repo,runtime,'SUCCESSOR_GPU_PILOT',error,track=track,completed_results=results)
    raise
finally:
    remaining=budget.remaining;budget.close()

with sqlite3.connect(budget_path) as db:
    charges=[{'id':row[0],'seconds':float(row[1]),'state':row[2]}
             for row in db.execute('SELECT id,seconds,state FROM compute_charges ORDER BY rowid')]
used=sum(row['seconds'] for row in charges)
state='SUCCEEDED_MEASURED_SUCCESSOR_PILOT' if len(results)==planned and all(r.get('state')=='SUCCEEDED' for r in results) else 'PARTIAL_MEASURED_SUCCESSOR_PILOT'
receipt={
    'state':state,'track':track,'information_set':info,'protocol_id':protocol['protocol_id'],'source_extension_protocol':'sgqx-v3.1.1-track-sources',
    'parent_study':'sgqx-v3','created_at':now(),'reserved_access':False,'qualified_for_final':False,
    'purpose':'cost_and_compatibility_measurement_not_model_selection',
    'planned_fits':planned,'completed_results':len(results),'total_two_track_fit_ceiling':pilot['pilot_fit_ceiling'],
    'pilot_gpu_seconds_ceiling_shared':pilot['pilot_gpu_seconds'],'charged_gpu_seconds_shared_ledger':used,
    'remaining_gpu_seconds_shared_ledger':remaining,'families':families,'year':year,'seed':seed,
    'fixed_settings':fixed,'results':results,'charges':charges,
    'execution_plan_id':plan_pointer['plan_id'],
    'source_tree_hash':code_hash(repo),'repair_attempt':retry_attempt,
    'manifest_id':q['manifest_id'],'parent_ledger':{k:v for k,v in parent.items() if k!='charges'},
    'protocol_sha256':file_hash(repo/'configs/completion_extension_v31.json'),
    'source_adr_sha256':file_hash(repo/'research/desktop_study/design_decisions/ADR-019-fred-segmentation-and-unit-stable-macro.md')
}
atomic_json(receipt_path,receipt)
if state=='SUCCEEDED_MEASURED_SUCCESSOR_PILOT':validate_pilot(repo,runtime,track)
else:preserve_incident(repo,runtime,'SUCCESSOR_GPU_PILOT',RuntimeError('Pilot compatibility failure; failed fits retained'),track=track,completed_results=results)
progress(repo,runtime,'SUCCESSOR GPU PILOT '+('MAIN' if track=='Main-A' else 'NESTED'),track=track,completed_fits=len(results),planned_fits=planned,status=state)
# Materialize a combined summary only after both independent track receipts exist.
other=repo/'reports'/('successor_gpu_pilot_'+('Nested_B' if track=='Main-A' else 'Main_A')+'.json')
if other.exists():
    other_doc=json.loads(other.read_text())
    combined={'state':'SUCCEEDED_BOTH_TRACK_MEASURED_SUCCESSOR_PILOTS' if other_doc.get('state')=='SUCCEEDED_MEASURED_SUCCESSOR_PILOT' and state=='SUCCEEDED_MEASURED_SUCCESSOR_PILOT' else 'PARTIAL',
        'created_at':now(),'reserved_access':False,'qualified_for_final':False,
        'receipts':{'Main-A':'reports/successor_gpu_pilot_Main_A.json','Nested-B':'reports/successor_gpu_pilot_Nested_B.json'}}
    atomic_json(repo/'reports/successor_gpu_pilot.json',combined)
print(json.dumps({k:v for k,v in receipt.items() if k not in {'results','charges'}},indent=2))
