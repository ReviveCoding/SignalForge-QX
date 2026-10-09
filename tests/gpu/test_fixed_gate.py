import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from signalforge.fixed_gate import FixedGateCUDA
from signalforge.runtime import gpu_lease


def test_retrained_fixed_gate_stays_uniform_and_reloads():
    runtime=Path(os.environ['SIGNALFORGE_RUNTIME'])
    rng=np.random.default_rng(11);x=rng.normal(size=(12,3,6)).astype(np.float32);y=x[:,-1,0]
    valid=np.ones((len(x),1),dtype=bool);valid[-3:]=False
    with gpu_lease(runtime),tempfile.TemporaryDirectory(dir=runtime/'tmp') as temp:
        model=FixedGateCUDA('rgmf_gru',width=8,epochs=3,base_columns=[0,1],source_columns=[[2,3]],meta_columns=[4,5])
        model.fit(x,y,1.,observed=np.ones_like(x,dtype=bool),source_valid=valid)
        assert not model.model.gate.weight.requires_grad and torch.count_nonzero(model.model.gate.weight)==0
        mean,q=model.predict(x,observed=np.ones_like(x,dtype=bool),source_valid=valid)
        with torch.no_grad():
            base_mean,base_q=model.model.base(torch.tensor(x,device='cuda')[...,[0,1]])
        assert np.array_equal(mean[-3:],base_mean.cpu().numpy()[-3:])
        assert np.array_equal(q[-3:],base_q.cpu().numpy()[-3:]) and np.all(np.diff(q)>=0)
        path=Path(temp)/'model.bin';model.save(path);restored=FixedGateCUDA.load(path)
        rm,rq=restored.predict(x,observed=np.ones_like(x,dtype=bool),source_valid=valid)
        assert np.array_equal(mean,rm) and np.array_equal(q,rq)
