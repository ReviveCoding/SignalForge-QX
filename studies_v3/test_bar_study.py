import numpy as np,torch,pytest
from studies_v3.bar_model import BAR,source_ood
from signalforge.v33_corrections import mature_partitions

def example():
    model=BAR([3,4],width=8);q=torch.tensor([[-2.,-1.,0.,1.,2.],[-2.,-1.,0.,1.,2.]],dtype=torch.float64)
    x=[torch.ones(2,5,3),torch.ones(2,5,4)];valid=torch.ones(2,2,dtype=torch.bool);ood=torch.zeros_like(valid);return model,q,x,valid,ood

def test_zero_init_exact_anchor():
    m,q,x,v,o=example();assert torch.equal(m(q,x,v,o),q)
def test_missing_ood_fallback_after_head_learning():
    m,q,x,v,o=example()
    for h in m.residual.heads:torch.nn.init.constant_(h.bias,.7)
    v[0]=False;o[1]=True;x[0][:]=float('nan');x[1][:]=float('nan')
    assert torch.equal(m(q,x,v,o),q)
def test_bounded_ordered_correction():
    m,q,x,v,o=example()
    for h in m.residual.heads:torch.nn.init.constant_(h.bias,20)
    new=m(q,x,v,o);assert (new[:,1:]>=new[:,:-1]).all();assert (new-q).abs().max()<=.15+1e-6
def test_invalid_active_source():
    m,q,x,v,o=example();x[0][0]=float('nan')
    with pytest.raises(ValueError):m(q,x,v,o)
def test_source_ood_any_context():
    o=np.zeros((2,3,5),dtype=bool);o[0,0,1]=True
    assert source_ood({'numeric_ood':o},[[0,1],[2,3]]).tolist()==[[True,False],[False,False]]
def test_mature_purge():
    dates=[f'2021-01-{d:02d}T00:00:00Z' for d in range(1,15)];ends=[f'2021-01-{d+2:02d}T00:00:00Z' for d in range(1,15)]
    parts=mature_partitions(dates,ends,ends,minimum=4,block=2)
    for p in parts:
        assert set(p['train']).isdisjoint(p['oof']);assert max(np.array(ends)[p['train']])<=p['cutoff'].replace('+00:00','Z')

def test_provider_clock_string_serialization():
    from studies_v3.runner import iso_clock
    import pandas as pd
    assert iso_clock('2021-01-01T22:00:00Z') == iso_clock(pd.Timestamp('2021-01-01T22:00:00Z'))
    with pytest.raises(Exception):iso_clock('invalid-clock')
