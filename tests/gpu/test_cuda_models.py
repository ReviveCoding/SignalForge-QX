"""Synthetic CUDA correctness, never public-data research evidence."""
import io
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from signalforge.runtime import gpu_lease
from signalforge.models import CudaTree
from signalforge.neural import Encoder,RGMF,normalized_loss,fit_encoder,require_cuda


@pytest.fixture(scope='module',autouse=True)
def exclusive_gpu():
    with gpu_lease(Path(os.environ['SIGNALFORGE_RUNTIME'])):
        require_cuda()
        torch.set_num_threads(4)
        yield


@pytest.mark.parametrize('family',['lightgbm','xgboost'])
def test_cuda_tree_objectives_and_reload(tmp_path,family):
    rng=np.random.default_rng(11)
    x=rng.normal(size=(128,8));y=x[:,0]+.1*rng.normal(size=128)
    m=CudaTree(family,rounds=4).fit(x,y)
    mean,q=m.predict(x)
    assert np.isfinite(mean).all() and np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
    m.save(tmp_path/'model')
    m2=m.load(tmp_path/'model')
    np.testing.assert_allclose(m2.predict(x)[0],mean,atol=1e-7)


@pytest.mark.parametrize('kind',['linear','mlp','gru','gru_d','tft','transformer'])
def test_encoder_finite_order_save_causal(kind):
    torch.manual_seed(11)
    model=Encoder(4,16,kind).cuda()
    x=torch.randn(16,6,4,device='cuda');y=x[:,-1,0]
    kw={'observed':torch.ones_like(x,dtype=torch.bool),'elapsed':torch.ones_like(x)} if kind=='gru_d' else {}
    mean,q=model(x,**kw)
    loss=normalized_loss(y,mean,q,1)
    loss.backward()
    assert torch.isfinite(loss) and torch.all(q[:,1:]>=q[:,:-1])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    other=Encoder(4,16,kind).cuda();other.load_state_dict(torch.load(buf,weights_only=True))
    torch.testing.assert_close(model(x,**kw)[0],other(x,**kw)[0],atol=0,rtol=0)
    assert model(torch.zeros(0,6,4,device='cuda'),**({k:v[:0] for k,v in kw.items()}))[0].shape==(0,)


@pytest.mark.parametrize('kind',['linear','gru','transformer'])
def test_rgfm_exact_fallback_and_masks(kind):
    model=RGMF(3,[2,2],4,16,kind).cuda()
    base=torch.randn(12,5,3,device='cuda')
    sources=[torch.randn(12,5,2,device='cuda') for _ in range(2)]
    meta=torch.randn(12,4,device='cuda');valid=torch.ones(12,2,dtype=torch.bool,device='cuda')
    valid[:4]=False;valid[4:8,0]=False
    mu,q,g=model(base,sources,meta,valid)
    mu0,q0=model.base(base)
    torch.testing.assert_close(mu[:4],mu0[:4],atol=0,rtol=0)
    torch.testing.assert_close(q[:4],q0[:4],atol=0,rtol=0)
    assert (g>=0).all() and torch.allclose(g.sum(1),torch.ones(12,device='cuda'))
    assert (g[:,1:][~valid]==0).all() and (q[:,1:]>=q[:,:-1]).all()
    normalized_loss(torch.randn(12,device='cuda'),mu,q,1).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_tiny_deterministic_recovery():
    torch.manual_seed(11)
    rng=np.random.default_rng(11)
    x=rng.normal(size=(32,4)).astype('float32');y=.5*x[:,0]-.2*x[:,1]
    model,losses=fit_encoder(Encoder(4,32,'mlp'),x,y,1,epochs=300,lr=.01)
    with torch.no_grad():mean,_=model(torch.tensor(x,device='cuda'))
    assert np.mean((mean.cpu().numpy()-y)**2)<.02
    assert losses[-1]<losses[0]*.25


def test_normalized_objective_units():
    y=torch.tensor([1.,-2.],device='cuda');m=y*.5;q=m[:,None].expand(-1,5)
    torch.testing.assert_close(normalized_loss(y,m,q,2),normalized_loss(y*100,m*100,q*100,200))


def test_fp32_bf16_reference():
    model=Encoder(4,32,'mlp').cuda();x=torch.randn(64,4,device='cuda')
    with torch.no_grad():
        fp=model(x)
        with torch.autocast('cuda',dtype=torch.bfloat16):bf=model(x)
    assert torch.cuda.is_bf16_supported()
    for a,b in zip(fp,bf):torch.testing.assert_close(a,b.float(),atol=.03,rtol=.03)
