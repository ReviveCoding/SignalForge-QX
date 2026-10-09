"""Retain native aborts that cannot reach a Python exception handler."""
import argparse,subprocess,sys,json
from signalforge.runtime import paths,now,atomic_json
from signalforge.successor_runtime import preserve_incident,progress
from signalforge.successor import require_successor_access
p=argparse.ArgumentParser();p.add_argument('--runner',choices=['pilot','full'],required=True);p.add_argument('--track',choices=['Main-A','Nested-B']);a=p.parse_args()
repo,runtime=paths();require_successor_access(repo)
script='run_successor_gpu_pilot.py' if a.runner=='pilot' else 'run_successor_gpu_full.py'
command=[sys.executable,str(repo/'scripts'/script)]
if a.track:command+=['--track',a.track]
log=runtime/'logs'/('supervised_successor_'+now().replace(':','')+'.log')
with log.open('w') as stream:
    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:
    text=log.read_text();incident_path=repo/'reports/successor_gpu_incident.json'
    prior=json.loads(incident_path.read_text()) if incident_path.exists() else {}
    if prior.get('state')!='OPEN':
        preserve_incident(repo,runtime,'SUCCESSOR_GPU_'+a.runner.upper(),RuntimeError(text[-12000:]),track=a.track,native_exit_code=result.returncode,worker_log=str(log),worker_traceback_available=result.returncode>0)
    progress(repo,runtime,'SUCCESSOR GPU '+a.runner.upper(),track=a.track,status='BLOCKED_PRESERVED_WORKER_FAILURE')
    raise SystemExit(1)
print(log.read_text())
