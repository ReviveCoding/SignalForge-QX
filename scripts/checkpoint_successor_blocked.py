"""Record actual continuation results and gates without claiming GPU completion."""
import json
from collections import Counter
from signalforge.runtime import paths,atomic_json,now,file_hash,digest,validate_bundle,commit_bundle,code_hash
from signalforge.successor import ledger_snapshot,assert_parent_unchanged,load_frozen_plan
from signalforge.completion import current_full_validation
from signalforge.successor_runtime import progress
repo,runtime=paths();reports=repo/'reports'
def read(name):return json.loads((reports/name).read_text(encoding='utf-8-sig'))
initial=read('successor_continuation_reconciliation_20261007.json')
parent=ledger_snapshot(runtime/'ledger/development_compute.sqlite',43200)
if parent['entries']!=3610 or parent['charges_digest']!='2705b3d5e54f99b4d5813cdab40a3de9ab9dfe849315f46f95075c47953748cb':
    raise PermissionError('INTEGRITY_OR_HASH_GATE: parent ledger changed')
counts=Counter();failures=[]
cpu={'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}
for directory in (runtime/'artifacts/track_development').iterdir():
    metric_path=directory/'metrics.json'
    if not metric_path.exists():continue
    row=json.loads(metric_path.read_text())
    if row.get('track') not in {'Main-A','Nested-B'} or row.get('family') not in cpu:continue
    validate_bundle(directory)
    if row.get('state')=='SUCCEEDED':
        model=runtime/row['model_relative_bundle']
        if digest(validate_bundle(model))!=row['model_receipt_id']:raise PermissionError('CPU immutable model differs')
        counts[row['track']]+=1
    else:failures.append(row)
if dict(counts)!={'Main-A':1925,'Nested-B':924} or failures:raise PermissionError('CPU artifact reconciliation differs')
plan,pointer=load_frozen_plan(repo,runtime)
pilot=ledger_snapshot(runtime/plan['pilot_ledger'],3600);full=ledger_snapshot(runtime/plan['full_ledger'],21600)
reason='Remaining weighted CUDA percentile boundary fails required regression; automatic repair ceiling 2/2 reached; additional backend build rejected by automatic approval review.'
block={'state':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE','reason':reason,'created_at':now(),
       'blocking_test':'tests/gpu/test_successor_weighted_cuda.py::test_weighted_cuda_quantile_tail_boundaries_use_sorted_support',
       'source_tree_hash':code_hash(repo),'reserved_access':False,'qualified_for_final':False}
atomic_json(reports/'successor_gpu_pilot_Nested_B.json',{**block,'track':'Nested-B','planned_fits':6,'completed_results':0,'results':[],'worker_started':False})
atomic_json(reports/'successor_gpu_pilot.json',{**block,'planned_fits':12,'successful_fits':0,
    'receipts':{'Main-A':'reports/successor_gpu_pilot_Main_A.json','Nested-B':'reports/successor_gpu_pilot_Nested_B.json'},'ledger':pilot})
atomic_json(reports/'successor_gpu_bill_admission.json',{**block,'admission_state':'NOT_ADMITTED_COMPATIBILITY_AND_VALIDATION_GATE',
    'execution_plan_id':pointer['plan_id'],'pilot_receipt_hashes':{t:file_hash(reports/('successor_gpu_pilot_'+t.replace('-','_')+'.json')) for t in plan['tracks']},
    'measured_gpu_seconds_by_track_family':{},'conservative_projected_full_study_seconds':None,
    'projection_unavailable_reason':'Neither complete six-family pilot exists; a single diagnostic is not a family bill',
    'frozen_plan_fit_counts':{n:p['planned_fit_count'] for n,p in plan['plans'].items()},
    'full_study_ledger_path':str(runtime/plan['full_ledger']),'full_gpu_seconds_ceiling':21600,
    'pilot_ledger':pilot,'parent_ledger_unchanged':True,'outcome_blind':True,'config_hashes':plan['config_hashes']})
for track in plan['tracks']:
    atomic_json(reports/('successor_gpu_full_'+track.replace('-','_')+'.json'),{**block,'track':track,'results':[],'worker_started':False})
atomic_json(reports/'successor_gpu_full.json',{**block,'tracks':[],'successful_fits':0,'worker_started':False,'ledger':full,'parent_ledger_unchanged':True})
calibration={t:{'state':read('calibration_'+t.replace('-','_')+'.json')['state'],
    'candidate_count':len(read('calibration_'+t.replace('-','_')+'.json')['candidates']),
    'receipt':'reports/calibration_'+t.replace('-','_')+'.json'} for t in plan['tracks']}
terminal={**block,'state':'MANUAL_BLOCKED_ADDITIONAL_RUNTIME_REPAIR_AUTHORIZATION',
    'branches':{'sources':'COMPLETED_RECONSTRUCTED_TIER_B','cpu_development':'COMPLETED','cpu_calibration':'COMPLETED_DEVELOPMENT_TIER_B',
                'cpu_statistical_analysis':'COMPLETED_DEVELOPMENT_TIER_B','successor_pilots':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE',
                'full_successor_study':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE','strict_pit':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE',
                'p1':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE','freeze':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE',
                'reserved_final':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE','prospective':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'},
    'successful_cpu_bundles':dict(counts),'total_successful_cpu_bundles':sum(counts.values()),'cpu_failure_artifacts':failures,
    'native_failed_pilot_fit_count':1,'successful_successor_pilot_fits':0,'successful_successor_full_fits':0,
    'parent_ledger':parent,'parent_ledger_unchanged':True,'pilot_ledger':pilot,'full_ledger':full,'calibration':calibration,
    'validation':current_full_validation(repo),'validation_receipt':'reports/test_execution.json',
    'execution_plan_id':pointer['plan_id'],'strict_gate_receipt':'reports/strict_pit_final_gate_audit.json',
    'scientific_authorization_exists':(repo/'.local/scientific_authorization.json').exists(),
    'freeze_receipt_exists':(repo/'.local/freeze_receipt.json').exists(),
    'implementation_complete':False,'reserved_evaluation_complete':False,'strict_final_claims_permitted':False,
    'automatic_repair_attempts':2,'maximum_automatic_repair_attempts':2,'gpu_pause_armed':(repo/'.local/PAUSE_BEFORE_GPU').exists(),
    'manual_review_patch':'scripts/build_successor_lightgbm_repair.py --attempt 2 --manual-boundary-review (prepared, NOT EXECUTED; separate manual authorization required; automatic ceiling stays 2)',
    'approval_review_rejection':'Additional backend modification/build exceeds the explicitly final second automatic repair attempt; no workaround permitted.'}
atomic_json(reports/'successor_continuation_terminal.json',terminal)
commit_bundle(runtime/'artifacts/successor_continuation_terminal'/digest(terminal),{'terminal.json':terminal},{'state':terminal['state'],'reserved_access':False})
atomic_json(reports/'AUTO_RECOVERY_STATE.json',{'state':'MANUAL_BLOCKED','stage':'CUDA_WEIGHTED_PERCENTILE_BOUNDARY','message':reason,
    'repair_attempt':2,'max_repair_attempts':2,'gpu_pause_armed':True,'observed_at':now(),'reserved_access':False})
progress(repo,runtime,'MANUAL BLOCKED / RESERVED FINAL SEALED',status=terminal['state'],completed_fits=0,planned_fits=12)
print(json.dumps({'state':terminal['state'],'cpu_bundles':dict(counts),'pilot_seconds':pilot['charged_seconds'],'parent_unchanged':True}))
