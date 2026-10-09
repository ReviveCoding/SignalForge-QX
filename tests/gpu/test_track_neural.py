import numpy as np
import torch
import os,pytest
from pathlib import Path
from signalforge.runtime import gpu_lease
from signalforge.track_neural import ExplicitSourceCUDA


@pytest.fixture(scope='module',autouse=True)
def exclusive_gpu():
    with gpu_lease(Path(os.environ['SIGNALFORGE_RUNTIME'])):
        from signalforge.neural import require_cuda
        require_cuda();torch.set_num_threads(4)
        yield


def test_training_raw_source_mask_exact_base_fallback_and_reload(tmp_path):
    x=np.random.default_rng(731).normal(size=(16,4,6)).astype('float32');y=x[:,-1,0]
    model=ExplicitSourceCUDA('rgmf_gru',width=8,epochs=3,base_columns=[0,1],source_columns=[[2,3]],meta_columns=[4,5])
    valid=np.zeros((len(x),1),dtype=bool);observed=np.ones_like(x,dtype=bool)
    model.fit(x,y,1.,source_valid=valid,observed=observed,checkpoint_path=tmp_path/'fit.pt')
    mean,q=model.predict(x,observed=observed,source_valid=valid)
    with torch.no_grad():base_mean,base_q=model.model.base(torch.as_tensor(x[...,[0,1]],device='cuda'))
    assert np.array_equal(mean,base_mean.cpu().numpy()) and np.array_equal(q,base_q.cpu().numpy())
    assert model.config['raw_source_mask_id']
    model.save(tmp_path/'model.pt');loaded=ExplicitSourceCUDA.load(tmp_path/'model.pt')
    lm,lq=loaded.predict(x,observed=observed,source_valid=valid)
    assert np.array_equal(mean,lm) and np.array_equal(q,lq)


def test_generic_multi_asset_cuda_fit_raw_reload_and_missing_source_mask(tmp_path):
    import pandas as pd
    from signalforge.runtime import paths,digest
    from signalforge.track_engine import prepare,_fit_registered
    from signalforge.experiments import ComputeBudget
    from signalforge.inference import infer_bundle
    repo,runtime=paths();dates=pd.date_range('2020-01-03',periods=44,freq='W-FRI',tz='UTC')
    frame=pd.DataFrame({'decision_time':dates.astype(str),'asset':['SPY']*44,'y':np.sin(np.arange(44)/7),
        'label_start':dates.astype(str),'label_end':(dates+pd.Timedelta(days=7)).astype(str),
        'label_available_at':(dates+pd.Timedelta(days=8)).astype(str),'max_dependency_available_at':dates.astype(str),
        'sequence_index':np.arange(44),'eligible':True,'context_eligible':True})
    frame.attrs['neural_contract']={'base_raw_columns':[0,1],'source_raw_columns':[[2,3]],'source_value_columns':[[2]],'meta_raw_columns':[4]}
    raw=np.random.default_rng(731).normal(size=(44,4,5));raw[:,:,2]=np.nan;raw[:,:,3:]=7.
    train=frame.iloc[:32];test=frame.iloc[34:];arrays=prepare(train,test,raw,'2020-08-20T00:00Z')
    assert not arrays['source_valid'].any() and not arrays['test_source_valid'].any()
    budget=ComputeBudget(tmp_path/'ledger/test_compute.sqlite',43200)
    try:result=_fit_registered(repo,tmp_path,arrays,test,'rgmf_gru',[8,.001],11,'Main-A','I3',2020,'outer','fixture_operator',digest(test[['decision_time','asset']].to_dict('records')),budget,'synthetic_fixture')
    finally:budget.close()
    assert result['state']=='SUCCEEDED' and result['evidence_kind']=='synthetic_fixture'
    mu,q=infer_bundle(tmp_path/result['model_relative_bundle'],raw[test.sequence_index],test.asset.tolist(),test.context_eligible.tolist())
    saved=np.load(tmp_path/'artifacts/track_development'/result['run_id']/'predictions.npz')
    np.testing.assert_allclose(mu,saved['mean'],atol=2e-6);np.testing.assert_allclose(q,saved['quantiles'],atol=2e-6)


def test_train_only_ssl_initialization_reused_across_downstream_trials(tmp_path):
    import pandas as pd
    from signalforge.runtime import paths
    from signalforge.track_engine import prepare
    from signalforge.track_ssl import frozen_ssl_initialization
    repo,_=paths();dates=pd.date_range('2020-01-03',periods=40,freq='W-FRI',tz='UTC')
    frame=pd.DataFrame({'decision_time':dates.astype(str),'asset':['a']*40,'y':np.sin(np.arange(40)/7),
        'label_start':dates.astype(str),'label_end':(dates+pd.Timedelta(days=7)).astype(str),
        'label_available_at':(dates+pd.Timedelta(days=8)).astype(str),'max_dependency_available_at':dates.astype(str),
        'sequence_index':np.arange(40),'eligible':True,'context_eligible':True})
    raw=np.random.default_rng(731).normal(size=(40,4,5));arrays=prepare(frame.iloc[:30],frame.iloc[32:],raw,'2020-08-20T00:00Z')
    state,identity,metadata=frozen_ssl_initialization(repo,tmp_path,arrays,8,11)
    receipt=tmp_path/'artifacts/track_ssl'/identity/'receipt.json';before=receipt.stat().st_mtime_ns
    reused,other,again=frozen_ssl_initialization(repo,tmp_path,arrays,8,11)
    assert identity==other and metadata==again and before==receipt.stat().st_mtime_ns
    for key in state:assert torch.equal(state[key],reused[key])


def test_generic_grud_actual_clock_decay_and_frozen_raw_reload(tmp_path):
    import pandas as pd
    from signalforge.runtime import paths,digest
    from signalforge.track_engine import prepare,_fit_registered
    from signalforge.experiments import ComputeBudget
    from signalforge.inference import infer_bundle
    repo,_=paths();dates=pd.date_range('2020-01-03',periods=44,freq='W-FRI',tz='UTC')
    frame=pd.DataFrame({'decision_time':dates.astype(str),'asset':['SPY']*44,'y':np.sin(np.arange(44)/7),
        'label_start':dates.astype(str),'label_end':(dates+pd.Timedelta(days=7)).astype(str),
        'label_available_at':(dates+pd.Timedelta(days=8)).astype(str),'max_dependency_available_at':dates.astype(str),
        'sequence_index':np.arange(44),'eligible':True,'context_eligible':True})
    frame['context_lineage']=[[{'decision_time':t.isoformat()} for t in pd.date_range(end=d,periods=4,freq='7D')] for d in dates]
    raw=np.random.default_rng(731).normal(size=(44,4,3));raw[:,1:3,1]=np.nan
    train,test=frame.iloc[:32],frame.iloc[34:];arrays=prepare(train,test,raw,'2020-08-20T00:00Z')
    assert np.all(arrays['elapsed'][:,2,1]==14)
    budget=ComputeBudget(tmp_path/'ledger/test_compute.sqlite',43200)
    try:result=_fit_registered(repo,tmp_path,arrays,test,'gru_d',[8,.001],11,'Main-A','I0',2020,'outer','fixture_operator',digest(test[['decision_time','asset']].to_dict('records')),budget,'synthetic_fixture')
    finally:budget.close()
    assert result['state']=='SUCCEEDED' and result['evidence_kind']=='synthetic_fixture'
    clocks=[[r['decision_time'] for r in rows] for rows in test.context_lineage]
    mu,q=infer_bundle(tmp_path/result['model_relative_bundle'],raw[test.sequence_index],test.asset.tolist(),test.context_eligible.tolist(),clocks)
    saved=np.load(tmp_path/'artifacts/track_development'/result['run_id']/'predictions.npz')
    np.testing.assert_allclose(mu,saved['mean'],atol=2e-6);np.testing.assert_allclose(q,saved['quantiles'],atol=2e-6)
    with pytest.raises(ValueError,match='context-clock'):infer_bundle(tmp_path/result['model_relative_bundle'],raw[test.sequence_index],test.asset.tolist(),test.context_eligible.tolist())


def test_multi_asset_residual_cuda_exact_missing_source_base_and_reload(tmp_path):
    import pandas as pd
    from signalforge.track_residual import TrackResidualCUDA
    dates=pd.date_range('2020-01-03',periods=60,freq='W-FRI',tz='UTC').repeat(2)
    assets=np.tile(['a','b'],60);raw=np.random.default_rng(731).normal(size=(120,4,5));raw[:,:,2]=np.nan;raw[:,:,3:]=7.
    frame=pd.DataFrame({'decision_time':dates.astype(str),'asset':assets,'y':np.sin(np.repeat(np.arange(60),2)/7)*np.where(assets=='a',1.,10.),
        'label_end':(dates+pd.Timedelta(days=7)).astype(str),'label_available_at':(dates+pd.Timedelta(days=8)).astype(str),
        'max_dependency_available_at':dates.astype(str)})
    contract={'base_raw_columns':[0,1],'source_raw_columns':[[2,3]],'source_value_columns':[[2]],'meta_raw_columns':[3,4]}
    model=TrackResidualCUDA(contract,width=8,epochs=3).fit(raw,frame,{'a':1.,'b':10.},'2021-03-08T00:00Z',checkpoint_path=tmp_path/'residual-checkpoint.pt')
    mean,q=model.predict(raw)
    baseline=model.base.predict(model.base_transform.transform(raw[...,[0,1]].reshape(len(raw),-1)))
    np.testing.assert_allclose(mean,baseline[0],atol=2e-6);np.testing.assert_allclose(q,baseline[1],atol=2e-6)
    assert model.oof_dates>=13
    model.save(tmp_path/'residual.pt');loaded=TrackResidualCUDA.load(tmp_path/'residual.pt');lm,lq=loaded.predict(raw)
    np.testing.assert_array_equal(mean,lm);np.testing.assert_array_equal(q,lq)
    resumed=TrackResidualCUDA(contract,width=8,epochs=3).fit(raw,frame,{'a':1.,'b':10.},'2021-03-08T00:00Z',checkpoint_path=tmp_path/'residual-checkpoint.pt')
    assert resumed.losses==model.losses
