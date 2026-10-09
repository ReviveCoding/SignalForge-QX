"""Read-only decision for the local GPU watcher. Never acquires data or reserved outcomes."""
import json
from signalforge.runtime import paths, code_hash, now
from signalforge.completion import successor_boundary,current_full_validation
from signalforge.successor import successor_authorized,load_frozen_plan
from signalforge.successor_queue import completed_full_receipt

repo,runtime=paths()
current_hash=code_hash(repo)
test_path=repo/'reports/test_execution.json'
validation={'present':test_path.exists(),'current_tree':False,'passed':False,'tests':0}
if test_path.exists():
    doc=json.loads(test_path.read_text())
    results=doc.get('results',[])
    hashes={r.get('source_tree_hash') for r in results if r.get('source_tree_hash')}
    validation['tests']=sum(int(r.get('counts',{}).get('tests',0) or 0) for r in results)
    validation['passed']=bool(results) and all(int(r.get('exit_code',1))==0 for r in results)
    validation['current_tree']=hashes=={current_hash}
    qualification=repo/'reports/track_input_qualification_v311.json'
    validation['newer_track_qualification']=bool(
        qualification.exists() and qualification.stat().st_mtime_ns>test_path.stat().st_mtime_ns
    )
    if validation['newer_track_qualification']:
        validation['passed']=False
else:
    validation['newer_track_qualification']=False

validation.update(current_full_validation(repo))

boundary=successor_boundary(repo)
pilot_script=repo/'scripts/run_successor_gpu_pilot.py'
pilot_receipts={
    'Main-A':repo/'reports/successor_gpu_pilot_Main_A.json',
    'Nested-B':repo/'reports/successor_gpu_pilot_Nested_B.json',
}
implementation_path=repo/'IMPLEMENTATION_STATUS.json'
implementation=json.loads(implementation_path.read_text()) if implementation_path.exists() else {}
full_complete=bool(implementation.get('implementation_complete')) and bool(implementation.get('reserved_evaluation_complete'))

incident_path=repo/'reports/successor_gpu_incident.json'
incident=json.loads(incident_path.read_text()) if incident_path.exists() else None
admission_path=repo/'reports/successor_gpu_bill_admission.json'
admission=json.loads(admission_path.read_text()) if admission_path.exists() else None
authorized=successor_authorized(repo)
full_path=repo/'reports/successor_gpu_full.json'
full_doc=json.loads(full_path.read_text()) if full_path.exists() else None
terminal_full=False
if full_doc and full_doc.get('state')=='SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT':
    frozen_plan,plan_pointer=load_frozen_plan(repo,runtime)
    terminal_full=completed_full_receipt(full_doc,admission,frozen_plan,plan_pointer['plan_id'])
if not authorized:
    action='WAIT_PREREQUISITES';reason='AUTH_GATE: registered successor development authorization absent.'
elif (repo/'.local/STOP_SUCCESSOR_GPU_RECOVERY').exists():
    action='WAIT_PREREQUISITES';reason='STOP_SUCCESSOR_GPU_RECOVERY control is armed.'
elif incident and incident.get('state')!='RESOLVED_VERIFIED':
    action='WAIT_PREREQUISITES';reason=incident.get('classification','UNKNOWN_UNSAFE')+': '+incident.get('reason','unresolved GPU incident')
elif full_complete:
    action='FULL_STUDY_COMPLETE';reason='Implementation and reserved evaluation are both complete.'
elif not validation['passed'] or not validation['current_tree']:
    action='RUN_VALIDATION';reason='Current source tree lacks a passing full source-bound validation receipt.'
elif terminal_full:
    action='IDLE_AFTER_PILOT';reason='Verified completed successor study is terminal; later source hashes require validation, not repeated bill admission or fits. Strict scientific/final gates remain separate.'
elif not pilot_script.exists():
    action='WAIT_SUCCESSOR_RUNNER';reason='Measured successor pilot runner is absent.'
else:
    action=None;reason=None
    resolved_repair=bool(incident and incident.get('state')=='RESOLVED_VERIFIED')
    pilot_docs={track:(json.loads(path.read_text()) if path.exists() else None) for track,path in pilot_receipts.items()}
    for track,token in [('Main-A','MAIN'),('Nested-B','NESTED')]:
        gate=boundary['track_boundaries'][track]
        receipt=pilot_docs[track]
        retryable=bool(receipt and receipt.get('state')!='SUCCEEDED_MEASURED_SUCCESSOR_PILOT' and resolved_repair)
        if gate['state']=='READY_FOR_MEASURED_SUCCESSOR_PILOT' and (receipt is None or retryable):
            action='RUN_SUCCESSOR_PILOT_'+token
            reason=('Verified repair resolved the preserved prior pilot failure; retrying '+track+'.'
                    if retryable else 'Scientific successor pilot boundary is ready for '+track+'.')
            break
    if action is None:
        successful=all(doc and doc.get('state')=='SUCCEEDED_MEASURED_SUCCESSOR_PILOT' for doc in pilot_docs.values())
        if successful:
            if not admission or (admission.get('state')=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE' and resolved_repair):
                action='RUN_SUCCESSOR_BILL_ADMISSION';reason='Both measured pilots succeeded; outcome-blind frozen bill admission required.'
            elif admission['state']!='ADMITTED_MEASURED_SUCCESSOR_BILL':
                action='IDLE_AFTER_PILOT';reason='RESOURCE_OR_BUDGET_GATE: neither frozen full-study plan fits the immutable successor ceiling.'
            elif admission.get('source_tree_hash')!=current_hash:
                action='RUN_SUCCESSOR_BILL_ADMISSION';reason='Verified repair changed source hash; fresh outcome-blind admission required.'
            else:
                full_path=repo/'reports/successor_gpu_full.json'
                full_doc=json.loads(full_path.read_text()) if full_path.exists() else None
                retry_full=bool(full_doc and full_doc.get('state')=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE' and resolved_repair)
                if full_doc is None or retry_full:
                    action='RUN_SUCCESSOR_FULL';reason=('Verified repair supersedes the preserved blocked full-study checkpoint.'
                        if retry_full else 'Frozen measured successor full-study bill admitted.')
                else:
                    action='IDLE_AFTER_PILOT';reason='Successor full-study terminal receipt exists; postprocessing and scientific gates remain separate.'
        else:
            pending=[track+': '+boundary['track_boundaries'][track].get('reason','blocked')
                     for track in ['Main-A','Nested-B'] if not pilot_docs[track]]
            action='WAIT_PREREQUISITES';reason='; '.join(pending) if pending else 'Terminal pilot incompatibility remains; no unbounded retry.'

print(json.dumps({
    'observed_at':now(),'action':action,'reason':reason,'validation':validation,
    'successor_boundary':boundary,'pilot_runner_present':pilot_script.exists(),
    'pilot_receipts_present':{k:v.exists() for k,v in pilot_receipts.items()},
    'reserved_access':False
},indent=2))
