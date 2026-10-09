"""Bounded exact-input CUDA memory-safety regression, charged to successor pilot."""
import json,subprocess,sys,time
from signalforge.runtime import paths,gpu_lease,atomic_json,now
from signalforge.successor_backend import activate_backend
from signalforge.experiments import ComputeBudget

repo,runtime=paths();backend=activate_backend(repo,runtime)
if backend is None:raise PermissionError('Private CUDA backend required')
child='''import numpy as np,lightgbm as lgb
from signalforge.runtime import paths
from signalforge.successor_backend import activate_backend
repo,runtime=paths()
print('CUDA weighted regression',flush=True)
data=np.load(runtime/'artifacts/successor_cuda_failure_inputs/Main_A_I3_2022.npz')
print(list(data.files),flush=True)
x=data['x'].astype(np.float32);y=data['y'].astype(np.float32);weights=data['weights']
model=lgb.train({'device_type':'cuda','objective':'quantile','alpha':0.05,'lambda_l2':1.0,'num_leaves':15,'max_bin':63,'min_data_in_leaf':10,'seed':11,'verbosity':-1,'num_threads':4},lgb.Dataset(x,label=y,weight=weights),num_boost_round=100)
prediction=model.predict(x[:8]);assert np.isfinite(prediction).all()
print('PASSED_EXACT_WEIGHTED_CUDA_INPUT',flush=True)
'''
# Activate before importing LightGBM in the isolated child.
child=child.replace('import numpy as np,lightgbm as lgb','import numpy as np').replace("repo,runtime=paths()","repo,runtime=paths();activate_backend(repo,runtime)\nimport lightgbm as lgb")
budget=ComputeBudget(runtime/'ledger/successor_gpu_pilot_compute.sqlite',3600)
started=time.monotonic()
try:
    with gpu_lease(runtime):
        from signalforge.resources import require_current_resources
        require_current_resources(repo,runtime)
        with budget.charge('private_weighted_cuda_regression_'+backend['backend_id'],120):
            result=subprocess.run([sys.executable,'-c',child],capture_output=True,text=True,timeout=115)
            report={'state':'PASSED_EXACT_WEIGHTED_CUDA_INPUT' if result.returncode==0 else 'FAILED_PRIVATE_CUDA_REGRESSION',
                'created_at':now(),'backend_id':backend['backend_id'],'exit_code':result.returncode,'seconds':time.monotonic()-started,
                'stdout':result.stdout,'stderr':result.stderr,'reserved_access':False,'qualified_for_final':False}
            atomic_json(repo/'reports/successor_cuda_backend_regression.json',report)
            if result.returncode:raise RuntimeError('Private CUDA regression failed; immutable failed charge retained')
finally:budget.close()
print(json.dumps(report))
