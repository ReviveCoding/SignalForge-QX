import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
from signalforge.models import NeuralCUDA
from signalforge.runtime import gpu_lease
from signalforge.checkpoint import load_checkpoint


def test_adapter_checkpoint_reuse_and_corruption():
    runtime=Path(os.environ['SIGNALFORGE_RUNTIME'])
    with gpu_lease(runtime),tempfile.TemporaryDirectory(dir=runtime) as temp:
        path=Path(temp)/'checkpoint.pt'
        rng=np.random.default_rng(11);x=rng.normal(size=(24,4,6)).astype('float32');y=x[:,-1,0]*.2
        mask=np.ones_like(x,dtype=bool)
        first=NeuralCUDA('gru',width=8,epochs=4).fit(x,y,1,observed=mask,checkpoint_path=path,checkpoint_seconds=0)
        second=NeuralCUDA('gru',width=8,epochs=4).fit(x,y,1,observed=mask,checkpoint_path=path)
        np.testing.assert_array_equal(first.predict(x)[0],second.predict(x)[0])
        assert len(second.losses)==4
        path.write_bytes(b'corrupt')
        with pytest.raises(RuntimeError,match='quarantined'):NeuralCUDA('gru',width=8,epochs=4).fit(x,y,1,observed=mask,checkpoint_path=path)
        assert not path.exists() and len(list(Path(temp).glob('quarantine-*')))==1


def test_actual_interrupted_optimizer_boundary_restores_rng_and_fullbatch_cursor(monkeypatch):
    import json,torch
    import signalforge.checkpoint as checkpoint
    runtime=Path(os.environ['SIGNALFORGE_RUNTIME'])
    with gpu_lease(runtime),tempfile.TemporaryDirectory(dir=runtime) as temp:
        path=Path(temp)/'interrupted.pt';rng=np.random.default_rng(731)
        x=rng.normal(size=(24,4,6)).astype('float32');y=x[:,-1,0]*.2
        mask=np.ones_like(x,dtype=bool)
        reference=NeuralCUDA('gru',width=8,epochs=4,seed=11,lr=.001).fit(x,y,1,observed=mask)
        original=checkpoint.save_checkpoint
        def interrupt_after_committed_step(path,state,identity):
            original(path,state,identity)
            if state['step']==2:raise RuntimeError('TEST_INTERRUPTED_AFTER_COMMITTED_BOUNDARY')
        monkeypatch.setattr(checkpoint,'save_checkpoint',interrupt_after_committed_step)
        with pytest.raises(RuntimeError,match='TEST_INTERRUPTED'):
            NeuralCUDA('gru',width=8,epochs=4,seed=11,lr=.001).fit(x,y,1,observed=mask,checkpoint_path=path,checkpoint_seconds=0)
        receipt=json.loads(path.with_suffix('.receipt.json').read_text())
        state=load_checkpoint(path,receipt['identity'])
        assert state['step']==2 and state['sample_cursor']==48 and len(state['losses'])==2
        assert set(state['rng'])=={'torch','cuda','numpy','python'}
        assert state['optimizer']['state'] and state['optimizer']['param_groups'][0]['lr']==.001
        # This adapter uses constant LR and a deterministic full batch, so no
        # separate scheduler or random sampling state exists beyond the cursor.
        monkeypatch.setattr(checkpoint,'save_checkpoint',original)
        torch.rand(10,device='cuda');np.random.random(10)
        resumed=NeuralCUDA('gru',width=8,epochs=4,seed=11,lr=.001).fit(x,y,1,observed=mask,checkpoint_path=path)
        assert resumed.losses==reference.losses
        np.testing.assert_array_equal(resumed.predict(x)[0],reference.predict(x)[0])
