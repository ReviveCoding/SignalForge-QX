"""Write a bounded bootstrap receipt, not a claim of completed research."""
from pathlib import Path
import argparse,hashlib,json,platform,datetime,os
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--runtime',required=True);a=p.parse_args()
r=Path(a.runtime);root=Path(a.repo)
s=json.loads((r/'logs/gpu_smoke.json').read_text())
if not s.get('qualified'):raise SystemExit('CUDA qualification missing')
d={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'code_repo':str(root),'runtime':str(r),'python':platform.python_version(),
 'status':'BOOTSTRAP_CUDA_SMOKE_QUALIFIED_NOT_RESEARCH_COMPLETE',
 'stack_spec_sha256':hashlib.sha256((root/'configs/stack_candidate.json').read_bytes()).hexdigest(),
 'gpu_smoke_sha256':hashlib.sha256((r/'logs/gpu_smoke.json').read_bytes()).hexdigest(),
 'real_data_experiments_executed':False,'broker_orders':False,'full_model_implementation':False}
pth=r/'bootstrap_receipt.json';tmp=pth.with_suffix('.partial');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,pth)
(root/'.local').mkdir(exist_ok=True)
(root/'.local/bootstrap_receipt.json').write_text(json.dumps(d,indent=2)+'\n')
