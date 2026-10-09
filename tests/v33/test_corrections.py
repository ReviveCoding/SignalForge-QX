import numpy as np
import pandas as pd
import pytest
import torch
from signalforge.v33_corrections import BaselineAnchoredResidual,AsymmetricSourceCorrection,asymmetric_quantiles,mature_partitions,select_inner_baseline
def sample():
    q=torch.tensor([[-2.,-1.,0.,1.,2.],[-1.,-.5,.2,1.,1.5]])
    return q,[torch.ones(2,3),torch.ones(2,2)],torch.full((2,2),.4),torch.ones(2,2,dtype=torch.bool),torch.zeros(2,2,dtype=torch.bool)
@pytest.mark.parametrize('kind',['bar','abc'])
def test_zero_init_exact_baseline(kind):
    q,x,w,v,o=sample()
    if kind=='bar':
        mean,out=BaselineAnchoredResidual([3,2])(q[:,2],q,x,w,v,o)
        assert torch.equal(mean,q[:,2])
    else:out=AsymmetricSourceCorrection([3,2])(q,x,w,v,o)
    assert torch.equal(out,q)
@pytest.mark.parametrize('kind',['bar','abc'])
def test_absence_ood_fallback_and_masked_nan(kind):
    q,x,w,v,o=sample(); v[:]=False;x[0][:]=float('nan')
    if kind=='bar':
        m=BaselineAnchoredResidual([3,2]);m.heads[0].bias.data.fill_(10);_,out=m(q[:,2],q,x,w,v,o)
    else:
        m=AsymmetricSourceCorrection([3,2]);m.heads[0].bias.data.fill_(10);out=m(q,x,w,v,o)
    assert torch.equal(out,q)
    v[:]=True;o[:]=True
    if kind=='bar':_,out=m(q[:,2],q,x,w,v,o)
    else:out=m(q,x,w,v,o)
    assert torch.equal(out,q)
def test_bar_bounded_ordered_and_saveload():
    q,x,w,v,o=sample();m=BaselineAnchoredResidual([3,2])
    m.heads[0].bias.data.copy_(torch.tensor([100.,100.,-100.,100.,-100.,100.]))
    mean,out=m(q[:,2],q,x,w,v,o)
    assert (out[:,1:]>=out[:,:-1]).all() and (mean-q[:,2]).abs().max()<=.25
    assert (out-q).abs().max()<=.25
    other=BaselineAnchoredResidual([3,2]);other.load_state_dict(m.state_dict())
    assert torch.equal(other(q[:,2],q,x,w,v,o)[1],out)
def test_abc_constraints_and_noncrossing():
    q,x,w,v,o=sample();shape=w.shape
    out=asymmetric_quantiles(q,torch.ones(shape),torch.full(shape,.5),torch.full(shape,.8),torch.full(shape,.1),w,v,o)
    assert (out[:,1:]>=out[:,:-1]).all()
    with pytest.raises(ValueError):asymmetric_quantiles(q,torch.ones(shape)*2,torch.zeros(shape),torch.ones(shape),torch.zeros(shape),w,v,o)
    with pytest.raises(ValueError):asymmetric_quantiles(q,torch.ones(shape),torch.ones(shape),torch.zeros(shape),torch.zeros(shape),w,v,o)
def test_abc_zero_rho_gradient_not_dead():
    q,x,w,v,o=sample();m=AsymmetricSourceCorrection([3,2]); out=m(q,x,w,v,o)
    out[:,4].sum().backward()
    assert m.heads[0].bias.grad[1]>0 and torch.isfinite(m.heads[0].bias.grad).all()
def test_invalid_gates_or_active_nan_fail_closed():
    q,x,w,v,o=sample();m=BaselineAnchoredResidual([3,2])
    with pytest.raises(ValueError):m(q[:,2],q,x,w*2,v,o)
    x[0][:]=float('nan')
    with pytest.raises(ValueError):m(q[:,2],q,x,w,v,o)
def test_prequential_purge_delayed_maturity_and_no_test_labels():
    d=pd.date_range('2019-01-01',periods=40,freq='7D',tz='UTC');ends=d+pd.Timedelta(days=7);a=d+pd.Timedelta(days=8)
    parts=mature_partitions(d,ends,a,10,5)
    for p in parts:
        cutoff=pd.Timestamp(p['cutoff']); idx=p['train']
        assert (d[idx]<cutoff).all() and (ends[idx]<=cutoff).all() and (a[idx]<=cutoff).all()
        assert set(idx).isdisjoint(p['oof'])
    delayed=a.to_numpy().copy();delayed[0]=d[-1]
    assert all(0 not in p['train'] for p in mature_partitions(d,ends,delayed,10,5)[:-1])
def test_inner_selection_rejects_outer_scores():
    assert select_inner_baseline([{'partition':'inner_validation','baseline_id':'a','normalized_pinball':.2},{'partition':'inner_validation','baseline_id':'b','normalized_pinball':.3}])=='a'
    with pytest.raises(PermissionError):select_inner_baseline([{'partition':'outer','baseline_id':'a','normalized_pinball':.001}])
def test_prototype_manifests_truthful():
    assert not BaselineAnchoredResidual([2]).manifest()['real_fit_executed']
    assert not AsymmetricSourceCorrection([2]).manifest()['eval_labels_accepted']
