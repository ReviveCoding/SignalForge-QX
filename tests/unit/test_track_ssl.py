import numpy as np
import pandas as pd
import pytest
from signalforge.track_ssl import frozen_ssl_initialization


def test_track_ssl_future_feature_is_rejected_before_cuda_and_cache(tmp_path):
    from types import SimpleNamespace
    training=pd.DataFrame({'decision_time':['2022-01-07T23:00Z'],'max_dependency_available_at':['2022-01-08T00:00Z']})
    prepared={'training':training,'transform':SimpleNamespace(fit_cutoff='2022-12-31T23:59Z'),'tx_sequence':np.ones((1,3,4))}
    with pytest.raises(PermissionError,match='future'):frozen_ssl_initialization(tmp_path,tmp_path,prepared,8,11)
    assert not (tmp_path/'artifacts').exists()
