import os
from pathlib import Path
import numpy as np
import pandas as pd
from signalforge.ssl import pretrain_gru
from signalforge.runtime import gpu_lease


def test_ssl_actual_cuda_train_only_features():
    runtime=Path(os.environ['SIGNALFORGE_RUNTIME']);rng=np.random.default_rng(11)
    x=rng.normal(size=(20,4,6)).astype('float32');dates=pd.date_range('2016-01-01',periods=20,freq='7D',tz='UTC')
    with gpu_lease(runtime):
        state=pretrain_gru(x,dates,dates,dates[-1],width=8,epochs=3)
    assert state['device']=='cuda:0' and np.isfinite(state['losses']).all()
    assert set(state['encoder_state']) and state['data_id']
