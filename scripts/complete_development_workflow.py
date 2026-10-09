"""Bounded continuation after the already active run; never overlaps GPU owners."""
import json,time,subprocess,sys
from pathlib import Path
import pandas as pd
from signalforge.runtime import paths,atomic_json,now,digest,code_hash,file_hash
from signalforge.reporting import render,validate_lineage
from signalforge.integrity import create_freeze,verify_freeze

repo,runtime=paths();active_path=repo/'reports/auxiliary_execution_state.json'
initial=json.loads(active_path.read_text())
if initial['state']!='RUNNING' and '--after-completed' not in sys.argv:
    raise RuntimeError('Expected the existing active development run; use --after-completed for a reconciled terminal checkpoint')
original=initial['started_at'];deadline=pd.Timestamp(original)+pd.Timedelta(hours=12)
state={'state':'WAITING_FOR_OWN_DEVELOPMENT_RUN','original_run_started_at':original,'started_at':now(),'stages':[],
       'full_study_complete':False,'final_access':False,'runner_source_sha256':file_hash(Path(__file__))}
state_path=repo/'reports/remaining_workflow_state.json';atomic_json(state_path,state)
while pd.Timestamp.now(tz='UTC')<deadline:
    active=json.loads(active_path.read_text())
    if active['started_at']!=original:raise RuntimeError('Active run changed; continuation refused')
    if active['state']!='RUNNING':break
    time.sleep(30)
else:
    state['state']='PAUSED_WAIT_DEADLINE';atomic_json(state_path,state);raise SystemExit(2)
state.update(state='RUNNING',development_terminal_state=active['state']);atomic_json(state_path,state)
logs=runtime/'artifacts/workflow_logs'/digest({'original':original,'started':state['started_at']});logs.mkdir(parents=True,exist_ok=True)
stages=[('canonical_cache_preflight',['scripts/verify_canonical_cache_upgrade.py','--require-complete'],900),
        ('canonical_materialization',['scripts/materialize_canonical_cache.py'],3600),
        ('outer_reload',['scripts/verify_auxiliary_models.py','--all'],900),
        ('frozen_outages',['scripts/run_cpu_diagnostics.py','--all'],2400),
        ('frozen_gate_controls',['scripts/run_frozen_gate_controls.py'],1200),
        ('invalid_CUDA_controls',['scripts/run_invalid_cuda_controls.py'],7200),
        ('capacity_comparison',['scripts/run_capacity_development.py'],7200),
        ('capacity_reload',['scripts/verify_capacity_models.py'],900),
        ('finalist_postprocessing',['scripts/run_cpu_postprocessing.py','--all'],9000),
        ('finalist_reload',['scripts/verify_cpu_postprocessing.py','--all'],900),
        ('temporal_transport',['scripts/run_temporal_transport.py','--all'],1800),
        ('research_figures',['scripts/render_research_figures.py'],180),
        ('real_systems',['scripts/benchmark_real_systems.py'],1200),
        ('experiment_status',['scripts/update_experiment_status.py'],120),
        ('training_import_boundary',['scripts/audit_training_imports.py'],120),
        ('full_validation',['scripts/run_validation.py','--real-source'],900),
        ('checkpoint',['scripts/update_checkpoint.py'],120)]
for name,arguments,timeout in stages:
    if name=='canonical_materialization' and state['stages'][-1]['state']!='SUCCEEDED':
        state['stages'].append({'stage':name,'state':'BLOCKED_CACHE_PROOF','reason':'Full exact-input preflight failed; no automatic retraining'});atomic_json(state_path,state);continue
    state['active_stage']=name;atomic_json(state_path,state)
    log=logs/(name+'.log');started=now();source_hash=code_hash(repo)
    try:
        with log.open('w') as stream:result=subprocess.run([sys.executable,*arguments],stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        entry={'stage':name,'exit_code':result.returncode,'state':'SUCCEEDED' if result.returncode==0 else 'BLOCKED_OR_FAILED','started_at':started,'ended_at':now(),'log':str(log)}
    except subprocess.TimeoutExpired:
        entry={'stage':name,'state':'FAILED_TIMEOUT','timeout_seconds':timeout,'started_at':started,'ended_at':now(),'log':str(log)}
    entry['source_tree_hash_at_start']=source_hash
    entry['source_tree_hash_at_end']=code_hash(repo)
    state['stages'].append(entry);atomic_json(state_path,state)
    print(json.dumps(entry),flush=True)
    # Continue independent branches; a failed research stage cannot become PASS.
    subprocess.run([sys.executable,'-m','signalforge.cli','report','--study','sgqx-v3','--validate-lineage'],check=True,timeout=120)
try:
    auth_path=repo/'.local/scientific_authorization.json'
    if not auth_path.exists():raise PermissionError('Explicit scientific authorization absent')
    receipt_path=repo/'.local/freeze_receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else create_freeze(repo,runtime)
    verify_freeze(receipt,repo,runtime,json.loads(auth_path.read_text()))
except (PermissionError,RuntimeError,FileNotFoundError) as error:
    state['final_gate']={'state':'BLOCKED_DATA','reason':str(error),'reserved_data_read':False}
else:
    from signalforge.final import execute_frozen
    try:
        state['final_result']=execute_frozen(repo,runtime,receipt_path);state['final_access']=True
    except Exception as error:
        state['final_access']=(runtime/'ledger/final_access.json').exists()
        state['final_gate']={'state':'FAILED_AUTHORIZED_BATCH','error_type':type(error).__name__,'reason':str(error),
                             'reserved_access_ledger_present':(runtime/'ledger/final_access.json').exists()}
state.update(state='CHECKPOINT_NOT_FULL_STUDY_COMPLETE',active_stage=None,ended_at=now())
atomic_json(state_path,state)
with (repo/'EXECUTION_PLAN.md').open('a') as stream:
    stream.write('\n\n## '+now()+' — bounded development continuation\n')
    for entry in state['stages']:stream.write('- '+entry['stage']+': '+entry['state']+'; evidence '+entry.get('log',entry.get('reason','No stage log'))+'\n')
    stream.write('- Final gate: '+json.dumps(state.get('final_gate',{'state':'EXECUTED_AUTHORIZED_BATCH'}))+'\n')
subprocess.run([sys.executable,'scripts/update_checkpoint.py'],check=True,timeout=120)
subprocess.run([sys.executable,'-m','signalforge.cli','report','--study','sgqx-v3','--validate-lineage'],check=True,timeout=120)
print(json.dumps(state),flush=True)
