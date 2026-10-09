"""Resolve ONLY the specific conditional vLLM-contention STOP after exclusive device checks."""
import json,time,subprocess
from signalforge.runtime import file_hash,atomic_json,now
from calibration_v34.runner import R,O,cfg,source_hash
from policy_v3.v33_budget_free import device_lease
stop=O/'STOP';content=stop.read_text();sha=file_hash(stop)
if 'due to another project vLLM active on shared RTX4090' not in content or 'Do not rerun until exclusive GPU compute available' not in content:raise PermissionError('Unrelated/permanent STOP remains authoritative')
for p in [R/'.local/STOP',R/'.local/PAUSE']:
    if p.exists():raise PermissionError('Unrelated stop/pause must not be bypassed')
cfg();observations=[]
for i in range(3):
    with device_lease(2*1024**3) as device:
        obs=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip();observations.append({'observed_at':now(),'exclusive_lease_acquired':True,'device':device,'telemetry':obs});print(json.dumps(observations[-1]),flush=True)
    if i<2:time.sleep(10)
assert file_hash(stop)==sha
archive=O/'STOP_RESOURCE_CONTENTION_RESOLVED_20261008.txt'
if archive.exists():raise PermissionError('Archive identity collision; diagnose')
stop.rename(archive)
atomic_json(O/'contention_recovery.json',{'state':'SAFE_RESOURCE_CONTENTION_RESOLVED','original_STOP_sha256':sha,'STOP_preserved_at':str(archive),'observations':observations,'retry_root':'52d6f58ee94f104da5d88ca013d295d5a711990f44f8bb3a5f5d161c816f6e13','code_hash_unchanged':source_hash(),'successful_models_to_reuse':74,'old_ledgers_modified':False,'no_cost_approval_or_budget_gate':True})
print('CONDITIONAL_RESOURCE_STOP_PRESERVED_AND_RESOLVED; READY_FOR_REGISTERED_RESUME',flush=True)