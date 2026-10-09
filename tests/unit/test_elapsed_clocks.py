import numpy as np
import pytest
from signalforge.track_neural import elapsed_from_context_clocks


def test_actual_context_clock_gap_and_dst_missingness_elapsed():
    raw=np.asarray([[[2.],[np.nan],[np.nan],[3.]]])
    times=[['2020-03-06T23:00Z','2020-03-13T22:00Z','2020-03-27T22:00Z','2020-04-03T22:00Z']]
    actual=elapsed_from_context_clocks(raw,times,[times[0][-1]])
    np.testing.assert_allclose(actual[0,:,0],[0,7-1/24,21-1/24,28-1/24])
    assert not actual[:,:,1].any()
    changed=raw.copy();changed[0,-1]=1e9
    np.testing.assert_array_equal(actual[:,:3],elapsed_from_context_clocks(changed,times)[:,:3])


def test_elapsed_clock_padding_naive_reversal_and_future_origin_rejected():
    raw=np.asarray([[[np.nan],[np.nan],[1.]]])
    valid=[[None,'2020-01-03T23:00Z','2020-01-10T23:00Z']]
    assert elapsed_from_context_clocks(raw,valid)[0,0,0]==0
    for times in [[[None,'2020-01-03','2020-01-10']],
                  [['2020-01-03T23:00Z',None,'2020-01-10T23:00Z']],
                  [[None,'2020-01-10T23:00Z','2020-01-03T23:00Z']]]:
        with pytest.raises(ValueError):elapsed_from_context_clocks(raw,times)
    with pytest.raises(ValueError,match='actual origin'):elapsed_from_context_clocks(raw,valid,['2020-01-03T23:00Z'])
