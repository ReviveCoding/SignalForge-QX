import json
import numpy as np
import pandas as pd
import pytest
from signalforge.development import trial_grid,preprocess,fit_artifact
from signalforge.experiments import ComputeBudget
from signalforge.runtime import commit_bundle,atomic_json
from signalforge.analysis import analyze_auxiliary


def test_registered_full_grid_outcome_blind():
    assert len(trial_grid('lightgbm'))==20 and len(trial_grid('gru'))==20
    assert len(trial_grid('historical'))==1
    assert trial_grid('gru')[-1]['neural']==[64,.01]


@pytest.mark.parametrize('clock_dtype',['strings','timestamps'])
def test_cpu_model_bundle_reused(tmp_path,monkeypatch,clock_dtype):
    x=np.arange(120,dtype=float).reshape(20,3,2)
    dates=pd.date_range('2016-01-01',periods=20,freq='7D',tz='UTC')
    frame=pd.DataFrame({'sequence_index':range(20),'decision_time':dates.astype(str),'y':np.arange(20)/20,
                        'label_available_at':(dates+pd.Timedelta(days=1)).astype(str)})
    if clock_dtype=='timestamps':frame['decision_time']=dates
    training=frame.iloc[:15];testing=frame.iloc[15:];cut=dates[15]-pd.Timedelta(hours=1)
    arrays=preprocess(x,training,testing,cut);budget=ComputeBudget(tmp_path/'budget.sqlite',100)
    args=(tmp_path,tmp_path,budget,'historical',trial_grid('historical')[0],11,training,testing,arrays,'fixture_id','fixture_code','fixture_data')
    metric=fit_artifact(*args)
    assert (tmp_path/'artifacts/auxiliary/fixture_id/model.bin').exists()
    from signalforge.inference import infer_bundle
    prediction=infer_bundle(tmp_path/'artifacts/auxiliary/fixture_id',x[15:])
    assert prediction[0].shape==(5,) and prediction[1].shape==(5,5)
    monkeypatch.setattr('signalforge.development.fit_predict',lambda *a,**k:pytest.fail('Verified model retrained'))
    assert fit_artifact(*args)==metric and budget.remaining==100
    budget.close()


def test_analysis_missing_seed_uses_registered_fallback(tmp_path):
    config={'families':['historical','lightgbm'],'seeds':[11,37,71],'outer_years':[2018]}
    rows=[];dates=pd.date_range('2018-01-05',periods=20,freq='7D',tz='UTC').astype(str).tolist()
    for family in config['families']:
        for seed in config['seeds']:
            if family=='lightgbm' and seed==37:continue
            identity=family+str(seed)
            prediction={'decision_time':dates,'y':[0.]*20,'mean':[0.]*20,'quantiles':[[-2,-1,0,1,2]]*20}
            commit_bundle(tmp_path/'artifacts/auxiliary'/identity,{'predictions.json':prediction},{'fixture':True})
            rows.append({'state':'SUCCEEDED','family':family,'seed':seed,'outer_year':2018,'run_id':identity,'scale':1.,'normalizer_id':'shared'})
    atomic_json(tmp_path/'reports/auxiliary_development_results.json',{'protocol':config,'results':rows})
    report=analyze_auxiliary(tmp_path,tmp_path)
    candidate=report['comparison_table'][1]
    assert candidate['n_unique_dates']==20 and candidate['fallback_seed_dates']==20
    assert candidate['native_seed_date_coverage']==pytest.approx(2/3)
    assert report['paired_contrasts'][0]['n_market_dates']==20
    assert all(v is None for v in report['confirmatory_Holm'].values())
def test_cache_only_admission_denied_before_any_training_without_guard(tmp_path):
    import pytest
    from signalforge.development import run_auxiliary
    with pytest.raises(PermissionError,match='no-training guard'):run_auxiliary(tmp_path,tmp_path,cache_only=True)
