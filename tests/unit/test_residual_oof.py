import numpy as np
import pandas as pd
import pytest
from signalforge.residual import mature_oof
from signalforge.models import prequential_base


def fixture():
    dates=pd.date_range('2016-01-01',periods=60,freq='7D',tz='UTC')
    x=np.arange(120).reshape(60,2)/120;y=x[:,0]**2
    return x,y,dates,dates+pd.Timedelta(days=3),dates+pd.Timedelta(days=4)


def test_oof_future_target_feature_revision_invariance():
    x,y,dates,end,available=fixture()
    mu,q=mature_oof(x,y,dates,end,available,10,5)
    changed_x=x.copy();changed_x[40:]+=99999
    changed_y=y.copy();changed_y[40:]-=99999
    other,_=mature_oof(changed_x,changed_y,dates,end,available,10,5)
    np.testing.assert_array_equal(mu[:40],other[:40])
    assert np.isnan(mu[:10]).all() and np.isfinite(mu[10:]).all()


def test_oof_delayed_labels_excluded():
    x,y,dates,end,available=fixture()
    available=available.to_numpy().copy();available[9]=dates[-1]
    first,_=mature_oof(x,y,dates,end,available,10,5)
    changed=y.copy();changed[9]=100000
    other,_=mature_oof(x,changed,dates,end,available,10,5)
    np.testing.assert_array_equal(first,other)
    assert np.isnan(first[10:15]).all()
    with pytest.raises(ValueError,match='publication'):prequential_base(x,y,np.arange(60),10,5)
