"""Execute only a frozen, measured-bill-admitted successor development grid."""
import json
import time
from signalforge.runtime import paths,atomic_json,now,code_hash,file_hash,gpu_lease,validate_bundle,digest
from signalforge.completion import current_full_validation
from signalforge.successor import require_successor_access,load_frozen_plan,ledger_snapshot,assert_parent_unchanged,conditional_family_gate
from signalforge.successor_inputs import prepared_track
from signalforge.successor_runtime import progress,preserve_incident
from signalforge.experiments import ComputeBudget
from signalforge.panels import track_folds
import signalforge.track_engine as engine
from signalforge.successor_backend import activate_backend,operator_id

repo,runtime=paths();require_successor_access(repo)
if not current_full_validation(repo)['passed']:raise PermissionError('BLOCKED_VALIDATION: full current-tree validation required')
plan,pointer=load_frozen_plan(repo,runtime)
admission_path=repo/'reports/successor_gpu_bill_admission.json';admission=json.loads(admission_path.read_text())
if admission.get('state')!='ADMITTED_MEASURED_SUCCESSOR_BILL' or admission.get('execution_plan_id')!=pointer['plan_id']:
    raise PermissionError('RESOURCE_OR_BUDGET_GATE: frozen measured bill not admitted')
for track,expected in admission['pilot_receipt_hashes'].items():
    if file_hash(repo/'reports'/('successor_gpu_pilot_'+track.replace('-','_')+'.json'))!=expected:raise PermissionError('Pilot admission hash changed')
if admission['source_tree_hash']!=code_hash(repo):raise PermissionError('INTEGRITY_OR_HASH_GATE: admission source tree changed')
mode=admission['admitted_plan'];design=plan['plans'][mode]
parent=admission['parent_ledger'];assert_parent_unchanged(runtime,parent)
budget=ComputeBudget(runtime/plan['full_ledger'],21600);study=json.loads((repo/'configs/study.json').read_text())
tracks=[];completed=0;planned=design['planned_fit_count'];original_fit=engine._fit_registered
backend=activate_backend(repo,runtime)
pilot_reused=[]

class ReuseOnlyBudget:
    def charge(self,identity,reservation):
        # Exact model/evaluation hashes must already exist. A miss cannot train.
        raise RuntimeError('BLOCKED_RESOURCE: exact pilot cache miss; use full-study recipe')

def instrumented_fit(*args,**kwargs):
    global completed
    family,regularization,seed,track,info,year=args[4:10]
    require_successor_access(repo)
    progress(repo,runtime,'FULL SUCCESSOR GPU STUDY',track=track,family=family,information_set=info,year=year,seed=seed,
             completed_fits=completed,planned_fits=planned,status='FITTING_OR_REUSING_DURABLE_BUNDLE')
    arguments=list(args);arguments[11]=operator_id(repo,study,track,family,backend)
    result=None
    if mode=='fixed_setting_development' and year==2022 and seed==11 and info==('I3' if track=='Main-A' else 'I4'):
        pilot=json.loads((repo/'reports'/('successor_gpu_pilot_'+track.replace('-','_')+'.json')).read_text())
        reuse=list(arguments)
        reuse[10]='successor_pilot_'+track.replace('-','_')+('_repair'+str(pilot['repair_attempt']) if pilot.get('repair_attempt') else '')
        reuse[13]=ReuseOnlyBudget();reuse[14]='successor_gpu_cost_pilot_v311'
        try:
            candidate=original_fit(*reuse,**kwargs)
            expected=next(row for row in pilot['results'] if row['family']==family)
            if candidate==expected and candidate['state']=='SUCCEEDED':
                result=candidate;pilot_reused.append(candidate['run_id'])
        except RuntimeError as error:
            if not str(error).startswith('BLOCKED_RESOURCE: exact pilot cache miss'):raise
    if result is None:result=original_fit(*arguments,**kwargs)
    completed+=1
    assert_parent_unchanged(runtime,parent)
    progress(repo,runtime,'FULL SUCCESSOR GPU STUDY',track=track,family=family,information_set=info,year=year,seed=seed,
             completed_fits=completed,planned_fits=planned,status=result['state'],last_durable_run_id=result['run_id'])
    return result

try:
    engine._fit_registered=instrumented_fit
    for track in plan['tracks']:
        panels,contexts,q=prepared_track(repo,runtime,track)
        gates={}
        for info,panel in panels.items():
            folds,blocked=track_folds(panel,study,track)
            if blocked:raise RuntimeError('SCIENTIFIC_OR_DATA_GATE: blocked full-study fold')
            gates[info]={str(year):conditional_family_gate(fold['train'],plan['conditional_gate']) for year,fold in folds.items()}
        # Conditional expansion requires a separately measured compatible bill;
        # it cannot silently consume the six-family admission.
        if all(g['state']=='ADMITTED_CONDITIONAL_FAMILY' for rows in gates.values() for g in rows.values()):
            atomic_json(repo/'reports'/('successor_conditional_gate_'+track.replace('-','_')+'.json'),{'gates':gates,'reserved_access':False})
            raise RuntimeError('RESOURCE_OR_BUDGET_GATE: conditional architecture needs its own measured compatible admission')
        atomic_json(repo/'reports'/('successor_conditional_gate_'+track.replace('-','_')+'.json'),{'state':'NOT_ADMITTED_CONDITIONAL_FAMILY','gates':gates,'reserved_access':False})
        if mode=='registered_hpo':
            result=engine.run_panel_development(repo,runtime,panels,contexts,study,track,plan['families'],budget=budget,evidence_kind='successor_gpu_development_v311')
        else:
            rows=[];operator=engine.track_operator_id(repo,study,track)
            arrays={info:engine.validate_panel(panel,contexts[info]) for info,panel in panels.items()}
            with gpu_lease(runtime):
                for year in plan['tracks'][track]['outer_years']:
                    folds={info:track_folds(panel,study,track)[0][year] for info,panel in panels.items()}
                    keys=[set(zip(f['train'].loc[f['train'].context_eligible,'decision_time'],f['train'].loc[f['train'].context_eligible,'asset'])) for f in folds.values()]
                    common=set.intersection(*keys);normalizer=None
                    for info in sorted(panels):
                        fold=folds[info];training=fold['train'];training=training[[key in common for key in zip(training.decision_time,training.asset)]]
                        prepared=engine.prepare(training,fold['test'],arrays[info],fold['outer_cutoff'])
                        if normalizer is not None and normalizer!=prepared['normalizer_id']:raise ValueError('Shared normalizer mismatch')
                        normalizer=prepared['normalizer_id']
                        for family in plan['families']:
                            for seed in plan['seeds']:
                                rows.append(instrumented_fit(repo,runtime,prepared,fold['test'],family,design['fixed_settings'][family],seed,track,info,year,
                                    'successor_fixed_outer',operator,q['grid_id'],budget,'successor_gpu_fixed_setting_development_v311'))
                        atomic_json(repo/'reports'/('successor_full_progress_'+track.replace('-','_')+'.json'),{'track':track,'results':rows,'execution_plan_id':pointer['plan_id'],'created_at':now(),'reserved_access':False})
            result={'track':track,'state':'SUCCEEDED_DEVELOPMENT_SOFTWARE' if all(r['state']=='SUCCEEDED' for r in rows) else 'PARTIAL','results':rows,'blocked_folds':[],
                    'claim_boundary':design['claim_boundary'],'reserved_access':False,'qualified_for_final':False}
        atomic_json(repo/'reports'/('successor_gpu_full_'+track.replace('-','_')+'.json'),result);tracks.append(result)
except Exception as error:
    preserve_incident(repo,runtime,'FULL_SUCCESSOR_GPU_STUDY',error,completed_fit_calls=completed,completed_tracks=tracks)
    raise
finally:
    engine._fit_registered=original_fit;budget.close()

for track in tracks:
    for row in track['results']:
        validate_bundle(runtime/'artifacts/track_development'/row['run_id']);validate_bundle(runtime/row['model_relative_bundle'])
ledger=ledger_snapshot(runtime/plan['full_ledger'],21600);assert_parent_unchanged(runtime,parent)
expected_outer=sum(len(t['information_sets'])*len(t['outer_years']) for t in plan['tracks'].values())*len(plan['families'])*len(plan['seeds'])
state='SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT' if sum(len(t['results']) for t in tracks)==expected_outer and all(t['state']=='SUCCEEDED_DEVELOPMENT_SOFTWARE' for t in tracks) and ledger['charged_seconds']<=21600 else 'PARTIAL_SUCCESSOR_GPU_DEVELOPMENT'
receipt={'state':state,'created_at':now(),'execution_plan_id':pointer['plan_id'],'admission_sha256':file_hash(admission_path),
         'plan_mode':mode,'planned_fits':planned,'completed_fit_calls':completed,'expected_outer_results':expected_outer,
         'exact_pilot_bundles_reused':pilot_reused,
         'tracks':tracks,'ledger':ledger,'parent_ledger_unchanged':True,'source_tree_hash':code_hash(repo),
         'reserved_access':False,'qualified_for_final':False,'economic_qualified':False}
atomic_json(repo/'reports/successor_gpu_full.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='tracks'},indent=2))
