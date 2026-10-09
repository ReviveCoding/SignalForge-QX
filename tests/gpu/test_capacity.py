import os
from pathlib import Path
import numpy as np
import torch,pytest
from signalforge.runtime import gpu_lease
from signalforge.capacity import CapacityMatchedCUDA

@pytest.fixture(scope='module',autouse=True)
def exclusive_gpu():
    with gpu_lease(Path(os.environ['SIGNALFORGE_RUNTIME'])):
        from signalforge.neural import require_cuda
        require_cuda();torch.set_num_threads(4);yield

@pytest.mark.parametrize('kind',['gru','transformer'])
def test_actual_cuda_capacity_control_mean_quantiles_checkpoint_reload(tmp_path,kind):
    x=np.random.default_rng(731).normal(size=(16,4,28)).astype('float32');y=x[:,-1,0]
    model=CapacityMatchedCUDA(kind,width=8,epochs=3).fit(x,y,1.,checkpoint_path=tmp_path/(kind+'.pt'))
    mu,q=model.predict(x);assert np.isfinite(mu).all() and np.isfinite(q).all() and (np.diff(q)>=0).all()
    assert model.capacity_spec['relative_parameter_gap']<=.01
    model.save(tmp_path/'model.bin');loaded=CapacityMatchedCUDA.load(tmp_path/'model.bin');lm,lq=loaded.predict(x)
    np.testing.assert_array_equal(mu,lm);np.testing.assert_array_equal(q,lq)
