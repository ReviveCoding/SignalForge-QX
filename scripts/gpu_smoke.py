"""Actual local synthetic CUDA capability checks. Not financial/model results.
Run on the user's machine. Controller never initializes CUDA; children do.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys, tempfile, time
from pathlib import Path

def torch_probe():
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable; CPU fallback is prohibited'
    assert torch.cuda.device_count()==1, 'Expected exactly one allocated CUDA device'
    name=torch.cuda.get_device_name(0)
    assert '4090' in name and 'Laptop' in name, f'Unexpected device: {name}'
    free,total=torch.cuda.mem_get_info(0)
    assert free >= 2*1024**3, 'Less than 2 GiB free; do not kill other processes'
    torch.set_num_threads(4);torch.manual_seed(31);torch.cuda.manual_seed_all(31)
    torch.cuda.set_per_process_memory_fraction(min(.75,(free-1024**3)/total),0)
    x=torch.randn(64,12,device='cuda'); y=x[:,0]*.25+x[:,1]*.1
    m=torch.nn.Sequential(torch.nn.Linear(12,32),torch.nn.GELU(),torch.nn.Linear(32,6)).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    qs=torch.tensor([.05,.1,.5,.9,.95],device='cuda')
    bf16=bool(torch.cuda.is_bf16_supported()); losses=[]
    for step in range(8):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16): z=m(x)
        z=z.float(); err=y[:,None]-z[:,1:]; pin=torch.maximum(qs*err,(qs-1)*err).mean()
        loss=((z[:,0]-y)**2).mean()+pin
        assert torch.isfinite(loss)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step();losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    assert next(m.parameters()).device.type=='cuda'
    return {'backend':'torch','version':torch.__version__,'runtime_cuda':torch.version.cuda,'device':name,'nominal_total_bytes':total,
      'initial_free_bytes':free,'bf16_smoke':bf16,'mean_and_pinball_backward':True,'synthetic_losses':losses,'qualified':True}

def lgb_probe():
    import lightgbm as lgb
    import numpy as np
    rng=np.random.default_rng(31); x=rng.normal(size=(256,12)).astype('float32'); y=(x[:,0]+rng.normal(size=256)*.1).astype('float32')
    checks=[]
    for objective,alpha in [('regression',None)]+[('quantile',q) for q in [.05,.1,.5,.9,.95]]:
        par={'device_type':'cuda','objective':objective,'verbosity':1,'num_threads':4,'num_leaves':7,'max_bin':31,'seed':31,'min_data_in_leaf':8}
        if alpha is not None:par['alpha']=alpha
        model=lgb.train(par,lgb.Dataset(x,label=y),num_boost_round=4)
        pred=model.predict(x);assert np.isfinite(pred).all();assert model.params['device_type']=='cuda'
        checks.append({'objective':objective,'alpha':alpha,'passed':True})
    return {'backend':'lightgbm','version':lgb.__version__,'device_type':'cuda','objective_checks':checks,'qualified':True}

def xgb_probe():
    import xgboost as xgb
    import numpy as np
    rng=np.random.default_rng(31); x=rng.normal(size=(256,12)).astype('float32'); y=(x[:,0]+rng.normal(size=256)*.1).astype('float32')
    checks=[]
    for objective in ['reg:squarederror','reg:quantileerror']:
        par={'tree_method':'hist','device':'cuda:0','objective':objective,'max_depth':3,'seed':31,'nthread':4}
        if objective=='reg:quantileerror':par['quantile_alpha']=[.05,.1,.5,.9,.95]
        model=xgb.train(par,xgb.QuantileDMatrix(x,label=y),num_boost_round=4)
        conf=json.loads(model.save_config());device=conf['learner']['generic_param']['device']
        assert device.startswith('cuda'),f'XGBoost silently fell back to {device}'
        pred=model.predict(xgb.DMatrix(x));assert np.isfinite(pred).all()
        checks.append({'objective':objective,'device':device,'passed':True})
    return {'backend':'xgboost','version':xgb.__version__,'objective_checks':checks,'qualified':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',choices=['torch','lightgbm','xgboost']);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.backend:
        try:d=globals()[{'torch':'torch_probe','lightgbm':'lgb_probe','xgboost':'xgb_probe'}[a.backend]]()
        except Exception as e:
            d={'backend':a.backend,'qualified':False,'error':f'{type(e).__name__}: {e}'}
        out.write_text(json.dumps(d,indent=2)+'\n');return 0 if d['qualified'] else 1
    # Inter-process flock lease across all three backend probes.
    import fcntl
    runtime=Path(os.environ.get('SIGNALFORGE_RUNTIME',str(out.parent.parent)))
    runtime.joinpath('locks').mkdir(parents=True,exist_ok=True)
    lock=open(runtime/'locks/gpu0.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('BLOCKED_GPU: project GPU lease already held')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=False)
    if processes.returncode or processes.stdout.strip():
        raise SystemExit('BLOCKED_GPU: external process inventory unavailable or device occupied; no processes killed')
    # Unique children prevent accepting a receipt from an earlier invocation.
    child_dir=Path(tempfile.mkdtemp(prefix='gpu-smoke-',dir=out.parent))
    children=[]
    for backend in ['torch','lightgbm','xgboost']:
        child=child_dir/f'{backend}.json';log=child_dir/f'{backend}.log'
        t=time.monotonic()
        try:
            with log.open('w') as f:
                r=subprocess.run([sys.executable,__file__,'--backend',backend,'--output',str(child)],stdout=f,stderr=subprocess.STDOUT,timeout=300,check=False)
            d=json.loads(child.read_text()) if child.exists() else {'backend':backend,'qualified':False,'error':'No child receipt'}
            d['exit_code']=r.returncode
            if r.returncode != 0 or d.get('backend') != backend:
                d['qualified']=False
                d['error']='Child failed or backend receipt mismatched'
        except subprocess.TimeoutExpired:d={'backend':backend,'qualified':False,'error':'300-second qualification bound exceeded'}
        d['elapsed_seconds']=time.monotonic()-t;children.append(d)
        if not d.get('qualified'):break
    result={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'fixture':'synthetic_CAPABILITY_ONLY','financial_research_result':False,
      'qualified':len(children)==3 and all(x.get('qualified') for x in children),'checks':children}
    tmp=out.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,out)
    print(json.dumps(result,indent=2));return 0 if result['qualified'] else 1
if __name__=='__main__':raise SystemExit(main())
