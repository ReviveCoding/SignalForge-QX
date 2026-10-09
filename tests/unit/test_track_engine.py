import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from signalforge.track_engine import asset_normalizer,prepare,per_date_weights,run_panel_development,validate_panel


def fixture():
    dates=pd.date_range('2019-10-04','2022-12-30',freq='W-FRI',tz='UTC')
    rows=[]
    for i,date in enumerate(dates):
        for asset,scale in [('a',1.),('b',10.)]:
            rows.append({'decision_time':date.isoformat(),'asset':asset,'y':scale*np.sin(i/7),
                'label_start':date.isoformat(),'label_end':(date+pd.Timedelta(days=7)).isoformat(),
                'label_available_at':(date+pd.Timedelta(days=8)).isoformat(),
                'max_dependency_available_at':date.isoformat(),'context_eligible':True,'eligible':True})
    frame=pd.DataFrame(rows);frame['sequence_index']=np.arange(len(frame))
    x=np.stack([np.array([[i%9,np.nan],[i%9+1,0.]]) for i in range(len(frame))])
    return frame,x


def test_shared_asset_scales_train_only_future_mutation_and_date_weights():
    frame,x=fixture();cutoff=pd.Timestamp('2021-01-01T00:00Z')
    train=frame[pd.to_datetime(frame.label_available_at,utc=True)<cutoff];test=frame.iloc[-12:]
    a=prepare(train,test,x,cutoff);scales,identity=asset_normalizer(train,cutoff)
    assert scales['b']==pytest.approx(10*scales['a'])
    changed=x.copy();changed[test.sequence_index]=1e8
    b=prepare(train,test,changed,cutoff)
    assert a['transform'].manifest()==b['transform'].manifest() and a['normalizer_id']==b['normalizer_id']==identity
    assert not np.array_equal(a['vx'],b['vx'])
    assert np.all(per_date_weights(train)==.5)
    with pytest.raises(ValueError,match='Immature'):asset_normalizer(frame,cutoff)


def test_nested_cpu_full_two_folds_shared_information_artifacts_and_resume(tmp_path):
    frame,x=fixture();study=json.loads((Path(__file__).parents[2]/'configs/study.json').read_text())
    first=run_panel_development(Path(__file__).parents[2],tmp_path,{'I0':frame,'I1':frame.copy()},{'I0':x,'I1':x+1},study,'Nested-B',['historical'])
    assert first['state']=='SUCCEEDED_DEVELOPMENT_SOFTWARE' and len(first['results'])==12
    assert {r['year'] for r in first['results']}=={2021,2022}
    for year in [2021,2022]:assert len({r['normalizer_id'] for r in first['results'] if r['year']==year})==1
    assert not first['qualified_for_final'] and first['evidence_kind']=='synthetic_fixture'
    paths=list((tmp_path/'artifacts/track_development').glob('*/receipt.json'));before={p:p.stat().st_mtime_ns for p in paths}
    second=run_panel_development(Path(__file__).parents[2],tmp_path,{'I0':frame,'I1':frame.copy()},{'I0':x,'I1':x+1},study,'Nested-B',['historical'])
    assert first==second and before=={p:p.stat().st_mtime_ns for p in paths}
    from signalforge.inference import infer_bundle
    for result in first['results'][:3]:
        test=frame[pd.to_datetime(frame.decision_time,utc=True).dt.year.eq(result['year'])]
        mean,q=infer_bundle(tmp_path/result['model_relative_bundle'],x[test.sequence_index],test.asset.tolist(),test.context_eligible.tolist())
        stored=np.load(tmp_path/'artifacts/track_development'/result['run_id']/'predictions.npz')
        np.testing.assert_allclose(mean,stored['mean'],atol=1e-12);np.testing.assert_allclose(q,stored['quantiles'],atol=1e-12)


def test_track_development_reserved_guard_precedes_target_access_and_cuda_budget(tmp_path):
    frame,x=fixture();frame.loc[0,'decision_time']='2024-01-01T00:00Z';frame['y']=frame.y.astype(object);frame.loc[0,'y']='do not inspect'
    with pytest.raises(PermissionError,match='Reserved'):validate_panel(frame,x)
    study=json.loads((Path(__file__).parents[2]/'configs/study.json').read_text())
    with pytest.raises(PermissionError,match='Measured CUDA'):run_panel_development(Path(__file__).parents[2],tmp_path,{'I0':frame},{'I0':x},study,'Nested-B',['lightgbm'])


def test_track_fit_operator_excludes_reporting_and_binds_math_environment(tmp_path):
    from signalforge.track_engine import track_operator_id
    source=tmp_path/'src/signalforge';source.mkdir(parents=True);reports=tmp_path/'reports';reports.mkdir()
    for name in ['track_engine','models','features','pit','neural','track_neural','ssl','track_ssl','residual','track_residual','track_statistics']:(source/(name+'.py')).write_text('original math')
    (reports/'environment_audit.json').write_text(json.dumps({'resolved_packages':{'torch':'2.7.1'},'python':'3.11','dispatcher_sha256':'a'*64}))
    study=json.loads((Path(__file__).parents[2]/'configs/study.json').read_text());before=track_operator_id(tmp_path,study,'Nested-B')
    (source/'reporting.py').write_text('new figure/report');assert track_operator_id(tmp_path,study,'Nested-B')==before
    (source/'models.py').write_text('changed training math');assert track_operator_id(tmp_path,study,'Nested-B')!=before


def test_registered_model_failure_retains_full_grid_with_historical_fallback(tmp_path,monkeypatch):
    from signalforge.track_engine import _fit_registered
    from signalforge.models import Statistical
    frame,x=fixture();training=frame.iloc[:80];testing=frame.iloc[80:90]
    prepared=prepare(training,testing,x,pd.Timestamp('2020-08-01T00:00Z'))
    original=Statistical.fit
    def fail_only_ridge(self,*args,**kwargs):
        if self.family=='ridge':raise RuntimeError('registered failure fixture')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Statistical,'fit',fail_only_ridge)
    result=_fit_registered(Path(__file__).parents[2],tmp_path,prepared,testing,'ridge',1.,11,'Nested-B','I0',2021,'outer','fixture-code','fixture-grid',None,'synthetic_fixture')
    assert result['state']=='FAILED_PERMANENT' and result['native_prediction_coverage']==0
    assert result['n_grid_rows']==len(testing) and result['fallback_rows']==len(testing)
    saved=np.load(tmp_path/'artifacts/track_development'/result['run_id']/'predictions.npz')
    assert len(saved['mean'])==len(testing) and np.isfinite(saved['quantiles']).all()
def test_cpu_track_budget_denial_precedes_fit(tmp_path):
    from signalforge.experiments import ComputeBudget
    frame,x=fixture();repo=Path(__file__).parents[2]
    study=json.loads((repo/'configs/study.json').read_text())
    budget=ComputeBudget(tmp_path/'ledger/budget.sqlite',1)
    try:
        with pytest.raises(RuntimeError,match='PAUSED_BUDGET'):
            run_panel_development(repo,tmp_path,{'I0':frame},{'I0':x},study,'Nested-B',['historical'],budget=budget)
        assert not list((tmp_path/'artifacts/track_models').glob('*/receipt.json'))
        assert budget.remaining==1
    finally:budget.close()
