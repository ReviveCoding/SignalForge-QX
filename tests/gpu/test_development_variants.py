"""Exercise the registered real-engine variant paths with small synthetic inputs."""
import json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from signalforge.development import preprocess,fit_artifact
from signalforge.runtime import gpu_lease
from signalforge.experiments import ComputeBudget
from signalforge.inference import infer_bundle

@pytest.mark.parametrize('family',['lgb_no_age','lgb_no_coverage','rgmf_residual_gru','ssl_gru'])
def test_variant_artifact_reload(family):
    runtime=Path(os.environ['SIGNALFORGE_RUNTIME'])
    rng=np.random.default_rng(31);x=rng.normal(size=(70,4,14))
    dates=pd.date_range('2016-01-01',periods=70,freq='7D',tz='UTC')
    frame=pd.DataFrame({'sequence_index':range(70),'decision_time':dates.astype(str),
          'y':x[:,-1,0]*.3+rng.normal(size=70)*.1,'label_end':(dates+pd.Timedelta(days=3)).astype(str),
          'label_available_at':(dates+pd.Timedelta(days=4)).astype(str),
          'max_dependency_available_at':(dates-pd.Timedelta(days=1)).astype(str)})
    train=frame.iloc[:65];test=frame.iloc[65:];cut=dates[65]-pd.Timedelta(hours=1)
    arrays=preprocess(x,train,test,cut)
    repo=Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(dir=runtime/'tmp') as temp,gpu_lease(runtime):
        root=Path(temp);budget=ComputeBudget(root/'budget.sqlite',1200)
        trial={'regularization':1.,'neural':[8,.001]}
        metric=fit_artifact(repo,root,budget,family,trial,11,train,test,arrays,'variant_fixture','fixture','fixture',raw_sequences=x)
        mu,q=infer_bundle(root/'artifacts/auxiliary/variant_fixture',x[65:])
        saved=json.loads((root/'artifacts/auxiliary/variant_fixture/predictions.json').read_text())
        np.testing.assert_allclose(mu,saved['mean'],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(q,saved['quantiles'],rtol=1e-6,atol=1e-6)
        assert metric['state']=='SUCCEEDED' and budget.remaining<1200
        budget.close()
