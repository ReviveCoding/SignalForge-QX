"""Fresh package environment reproduction of the declared Tier-B CPU evidence.

Reuses the configured WSL Python interpreter, not its site-packages. It does not
qualify a fresh CUDA environment or the unavailable complete research study.
"""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
from importlib.metadata import version
from signalforge.runtime import paths,atomic_json,digest,file_hash,code_hash,commit_bundle,validate_bundle,now

PACKAGES=['numpy','pandas','scipy','scikit-learn','joblib','cloudpickle','threadpoolctl','python-dateutil','pytz','tzdata','six']

def verify(destination):
    import numpy as np
    from signalforge.data import auxiliary_rows
    from signalforge.auxiliary import sequences
    from signalforge.inference import infer_bundle
    from signalforge.metrics import quantile_loss
    from signalforge.models import QUANTILES
    repo,runtime=paths();destination=Path(destination).resolve()
    if not destination.is_relative_to(runtime/'artifacts/fresh_cpu_reproduction'):raise PermissionError('Project-local reproduction required')
    prefix=(destination/'environment').resolve()
    if Path(sys.prefix).resolve()!=prefix or sys.prefix==sys.base_prefix:raise PermissionError('Fresh virtual environment required')
    forbidden=(runtime/'env/lib/python3.11/site-packages').resolve()
    if any(Path(p).resolve()==forbidden for p in sys.path if p):raise PermissionError('Original environment site-packages leaked')
    protocol=json.loads((destination/'protocol.json').read_text())
    versions={p:version(p) for p in PACKAGES}
    if versions!=protocol['versions'] or code_hash(repo)!=protocol['code_hash']:raise ValueError('Fresh reproduction lock drift')
    acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
    frame,raw=sequences(auxiliary_rows(acquisition['records']))
    index={str(d):i for i,d in enumerate(frame.decision_time)}
    core=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
    checks=[]
    for row in core['results']:
        if row['family'] not in {'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}:continue
        directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
        saved=json.loads((directory/'predictions.json').read_text())
        # Date strings are normalized rather than relying on serialization style.
        import pandas as pd
        lookup={pd.Timestamp(d).isoformat():i for d,i in index.items()}
        x=raw[[lookup[pd.Timestamp(d).isoformat()] for d in saved['decision_time']]]
        mean,q=infer_bundle(directory,x)
        delta=max(float(np.max(np.abs(mean-np.asarray(saved['mean'])))),float(np.max(np.abs(q-np.asarray(saved['quantiles'])))))
        if delta>1e-10:raise ValueError('Fresh environment real CPU forecast parity failed')
        original=quantile_loss(np.asarray(saved['y']),np.asarray(saved['quantiles']),QUANTILES,row['scale'])
        recomputed=quantile_loss(np.asarray(saved['y']),q,QUANTILES,row['scale'])
        loss_delta=float(np.max(np.abs(original-recomputed)))
        if loss_delta>1e-12:raise ValueError('Fresh environment scoring parity failed')
        checks.append({'run_id':row['run_id'],'family':row['family'],'receipt_id':digest(receipt),
            'max_prediction_abs_delta':delta,'max_loss_abs_delta':loss_delta,'n_dates':len(x)})
    if len(checks)!=75 or len({c['family'] for c in checks})!=5:raise ValueError('Complete declared CPU evidence required')
    result={'state':'PASSED_FRESH_PACKAGE_ENVIRONMENT_TIER_B_CPU_REPRODUCTION','checks':checks,
        'fresh_site_packages':True,'configured_WSL_interpreter_reused':True,'new_model_fits':0,'CUDA_allocations':0,
        'versions':versions,'python_executable':sys.executable,'sys_prefix':sys.prefix,'code_hash':code_hash(repo),
        'reserved_access':False,'complete_research_reproduction':False,'created_at':now(),
        'claim_boundary':'75 actual Tier-B CPU outer bundles and scores reproduced in fresh isolated package environment; no fresh CUDA or full-study reproduction claim'}
    atomic_json(destination/'worker_result.json',result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify');args=parser.parse_args()
    if args.verify:return verify(args.verify)
    repo,runtime=paths()
    protocol={'versions':{p:version(p) for p in PACKAGES},'code_hash':code_hash(repo),
        'core_report_sha256':file_hash(repo/'reports/auxiliary_development_results.json'),
        'source_report_sha256':file_hash(repo/'reports/eia_development_acquisition.json'),
        'evidence_tier':'Tier-B CPU cached-model and score reproduction','max_wall_seconds':600,
        'new_fits':0,'reserved_access':False,'full_study_reproduction':False}
    destination=runtime/'artifacts/fresh_cpu_reproduction'/digest(protocol)
    if (destination/'receipt.json').exists():
        validate_bundle(destination);result=json.loads((destination/'result.json').read_text())
        atomic_json(repo/'reports/fresh_cpu_reproduction.json',result);return
    if shutil.disk_usage(runtime).free<2*1024**3:raise RuntimeError('BLOCKED_RESOURCE: fresh environment disk admission')
    destination.mkdir(parents=True,exist_ok=True);atomic_json(destination/'protocol.json',protocol)
    atomic_json(repo/'reports/fresh_cpu_reproduction_plan.json',protocol)
    env=destination/'environment';log=destination/'execution.log'
    local=dict(os.environ,PYTHONNOUSERSITE='1',PIP_CONFIG_FILE=os.devnull)
    try:
        with log.open('a') as stream:
            subprocess.run([sys.executable,'-m','venv','--copies',str(env)],check=True,timeout=60,stdout=stream,stderr=subprocess.STDOUT,env=local)
            python=str(env/'bin/python')
            subprocess.run([python,'-m','pip','--isolated','install','--only-binary=:all:','--no-deps',
                '--index-url','https://pypi.org/simple','--cache-dir',str(runtime/'cache/pip'),
                '--timeout','20','--retries','1','--report',str(destination/'install_report.json'),
                *[p+'=='+v for p,v in protocol['versions'].items()]],check=True,timeout=300,stdout=stream,stderr=subprocess.STDOUT,env=local)
            subprocess.run([python,'-m','pip','check'],check=True,timeout=30,stdout=stream,stderr=subprocess.STDOUT,env=local)
            subprocess.run([python,str(Path(__file__).resolve()),'--verify',str(destination)],cwd=repo,check=True,timeout=120,
                stdout=stream,stderr=subprocess.STDOUT,env=local)
        install=json.loads((destination/'install_report.json').read_text())
        for item in install['install']:
            from urllib.parse import urlsplit
            if urlsplit(item['download_info']['url']).hostname!='files.pythonhosted.org':raise PermissionError('Unexpected package source')
            if len(item['download_info']['archive_info']['hashes']['sha256'])!=64:raise ValueError('Package hash absent')
        result=json.loads((destination/'worker_result.json').read_text())
        result.update(protocol_id=digest(protocol),install_report_sha256=file_hash(destination/'install_report.json'))
        commit_bundle(destination,{'result.json':result,'frozen_protocol.json':protocol,
            'package_install_report.json':install,'reproduction_log.txt':log.read_bytes()},{'protocol_id':digest(protocol)})
        result['artifact_id']=digest(validate_bundle(destination))
    except Exception as error:
        result={'state':'BLOCKED_OR_FAILED_FRESH_CPU_REPRODUCTION','reason':str(error),'protocol':protocol,
            'partial_environment':str(env),'log':str(log),'reserved_access':False,'created_at':now()}
        commit_bundle(destination/'failed_attempts'/digest(result),{'failure.json':result},{'protocol_id':digest(protocol)})
    atomic_json(repo/'reports/fresh_cpu_reproduction.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if result['state']!='PASSED_FRESH_PACKAGE_ENVIRONMENT_TIER_B_CPU_REPRODUCTION':raise SystemExit(2)

if __name__=='__main__':main()
