"""The pilot crash was on weighted quantiles, unlike unweighted smoke tests."""
import subprocess
import sys
from signalforge.runtime import paths,gpu_lease

def test_successor_private_cuda_weighted_quantile_subprocess():
    repo,runtime=paths()
    program='''
import numpy as np
from signalforge.runtime import paths
from signalforge.successor_backend import activate_backend
repo,runtime=paths()
backend=activate_backend(repo,runtime)
assert backend is not None
import lightgbm as lgb
rng=np.random.default_rng(11)
x=rng.normal(size=(4576,8)).astype(np.float32)
y=rng.normal(size=4576).astype(np.float32)
weights=np.full(4576,0.125)
m=lgb.train({'device_type':'cuda','objective':'quantile','alpha':0.05,'lambda_l2':1.0,'num_leaves':15,'max_bin':63,'min_data_in_leaf':10,'num_threads':4,'verbosity':-1,'seed':11},lgb.Dataset(x,label=y,weight=weights),num_boost_round=4)
assert m.params['device_type']=='cuda'
assert np.isfinite(m.predict(x[:8])).all()
'''
    with gpu_lease(runtime):
        result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr

def test_weighted_cuda_quantile_tail_boundaries_use_sorted_support():
    repo,runtime=paths()
    program='''
import numpy as np
from signalforge.runtime import paths
from signalforge.successor_backend import activate_backend
repo,runtime=paths();activate_backend(repo,runtime)
import lightgbm as lgb
x=np.zeros((4,1),dtype=np.float32);y=np.array([2,3,4,5],dtype=np.float32);w=np.array([4,3,2,1],dtype=np.float32)
# Preserve the CUDA interior reversed-CDF interpolation: 3-(9.5-6)/(10-6)=2.125.
# The upper boundary must return the largest sorted support without index -1.
for alpha,expected in [(0.05,2.125),(0.95,5.)]:
 model=lgb.train({'device_type':'cuda','objective':'quantile','alpha':alpha,'lambda_l2':1.0,'num_leaves':15,'max_bin':63,'min_data_in_leaf':10,'seed':11,'verbosity':-1,'num_threads':4},lgb.Dataset(x,label=y,weight=w),num_boost_round=1)
 prediction=model.predict(x)
 assert model.params['device_type']=='cuda'
 assert np.allclose(prediction,expected,rtol=1e-6),(alpha,prediction,expected)
'''
    with gpu_lease(runtime):result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
