import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from signalforge.residual import ResidualRGMFCUDA
from signalforge.runtime import gpu_lease


@pytest.mark.parametrize('kind',['linear','gru','transformer'])
def test_mature_residual_cuda_reload_fallback(tmp_path,kind):
    rng=np.random.default_rng(11);x=rng.normal(size=(60,4,6)).astype('float32')
    y=(x[:,-1,0]*.3+rng.normal(size=60)*.1).astype('float32')
    dates=pd.date_range('2016-01-01',periods=60,freq='7D',tz='UTC')
    end=dates+pd.Timedelta(days=3);available=dates+pd.Timedelta(days=4)
    observed=np.ones_like(x,dtype=bool)
    with gpu_lease(Path(os.environ['SIGNALFORGE_RUNTIME'])):
        adapter=ResidualRGMFCUDA([[0,1,2]],[3,4,5],base_columns=[3,4],kind=kind,width=8,epochs=2)
        adapter.fit(x,y,1.,dates,end,available,available[-1],observed,minimum=10,block=5)
        mean,q=adapter.predict(x,observed)
        assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
        adapter.save(tmp_path/'model.pt');loaded=ResidualRGMFCUDA.load(tmp_path/'model.pt')
        np.testing.assert_array_equal(loaded.predict(x,observed)[0],mean)
        empty=np.zeros_like(observed)
        fallback=adapter.predict(x,empty)
        base=adapter.base.predict(adapter.transform.transform(x[...,[3,4]].reshape(len(x),-1)))
        np.testing.assert_allclose(fallback[0],base[0],rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(fallback[1],base[1],rtol=1e-7,atol=1e-7)
        assert adapter.oof_count==50 and adapter.device=='cuda:0'
