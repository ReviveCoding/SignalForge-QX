import numpy as np
import pandas as pd
import pytest
from signalforge.statistics import calendar_indices,paired_statistics


def test_shared_blocks_do_not_cross_calendar_gaps():
    dates=list(pd.date_range('2016-01-01',periods=20,freq='7D',tz='UTC'))+list(pd.date_range('2017-01-01',periods=20,freq='7D',tz='UTC'))
    indices,metadata=calendar_indices(dates,8,2000)
    assert metadata['segment_lengths']==[20,20]
    assert (indices[:,:20]<20).all() and (indices[:,20:]>=20).all()
    left=np.linspace(0,1,40);right=left-.02
    result=paired_statistics(left,right,indices,metadata)
    assert result['mean_improvement']==pytest.approx(.02)
    other,_=calendar_indices(dates,8,2000);np.testing.assert_array_equal(indices,other)
