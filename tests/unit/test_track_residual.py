import numpy as np
import pandas as pd
from signalforge.track_residual import native_oof_base


def test_multi_asset_oof_uses_past_native_target_scales_and_transform_only():
    dates=pd.date_range('2020-01-03',periods=60,freq='W-FRI',tz='UTC').repeat(2)
    assets=np.tile(['a','b'],60);x=np.random.default_rng(731).normal(size=(120,3,4));x[::3,:,1]=np.nan
    y=np.sin(np.repeat(np.arange(60),2)/7)*np.where(assets=='a',1.,10.)
    ends=dates+pd.Timedelta(days=7);available=ends+pd.Timedelta(hours=1)
    first,_=native_oof_base(x,y,assets,dates,ends,available,[0,1],minimum=10,block=5)
    changed=x.copy();changed[80:]=1e5;changed_y=y.copy();changed_y[80:]=-1e6
    other,_=native_oof_base(changed,changed_y,assets,dates,ends,available,[0,1],minimum=10,block=5)
    np.testing.assert_array_equal(first[:80],other[:80])
    assert np.isnan(first[:30]).all() and np.isfinite(first[30:]).all()
