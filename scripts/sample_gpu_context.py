"""Read-only local GPU telemetry, with unavailable WSL sensors explicit."""
import csv,io,json,subprocess
from signalforge.runtime import paths,atomic_json,now
repo,runtime=paths()
fields=['name','uuid','memory.total','memory.used','utilization.gpu','temperature.gpu','power.draw','power.limit','pstate','clocks.sm']
result=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
if result.returncode:raise RuntimeError('GPU telemetry unavailable; no fabricated values')
rows=list(csv.reader(io.StringIO(result.stdout)))
if len(rows)!=1:raise RuntimeError('Expected one physical GPU')
active=repo/'reports/auxiliary_execution_state.json'
record={'observed_at':now(),'sensors':dict(zip(fields,[v.strip() for v in rows[0]])),'physical_gpus':1,
    'sampler_allocates_CUDA':False,'concurrent_development_state':json.loads(active.read_text()).get('state') if active.exists() else 'UNKNOWN',
    'sensor_missingness':'N/A and unsupported values retained','source':'actual nvidia-smi read-only telemetry'}
path=repo/'reports/gpu_runtime_context.json';old=json.loads(path.read_text()) if path.exists() else {'observations':[]}
old['observations'].append(record);atomic_json(path,old);print(json.dumps(record))
