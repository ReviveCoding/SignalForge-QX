import os
from pathlib import Path
import numpy as np
import pytest
from signalforge.models import NeuralCUDA
from signalforge.runtime import gpu_lease


@pytest.mark.parametrize('kind',['mlp','gru','rgmf_linear','rgmf_gru','rgmf_transformer'])
def test_neural_adapter_save_predict(tmp_path,kind):
    with gpu_lease(Path(os.environ['SIGNALFORGE_RUNTIME'])):
        rng=np.random.default_rng(11);x=rng.normal(size=(24,4,8)).astype('float32');y=x[:,-1,0]*.2
        args={'base_columns':[0,1],'source_columns':[[2,3]],'meta_columns':[4,5,6,7]} if kind.startswith('rgmf') else {}
        model=NeuralCUDA(kind=kind,width=16,epochs=3,**args).fit(x,y,1,observed=np.ones_like(x,dtype=bool))
        before=model.predict(x,observed=np.ones_like(x,dtype=bool))
        model.save(tmp_path/'model.pt');loaded=NeuralCUDA.load(tmp_path/'model.pt')
        after=loaded.predict(x,observed=np.ones_like(x,dtype=bool))
        np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
