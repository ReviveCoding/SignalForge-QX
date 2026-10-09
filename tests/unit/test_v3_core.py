import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from signalforge.runtime import atomic_json, commit_bundle, validate_bundle, admission
from signalforge.sources import allowed_url,redact_url,validate_payload,discover_zip,nport_reference_months
from signalforge.features import TrainTransform, shared_target_scale, purged_training, dependency_closure
from signalforge.models import Statistical,prequential_base
from signalforge.evaluation import score_grid,holm,compare
from signalforge.postprocess import Calibrator
from signalforge.portfolio import Ledger,feasible_weights
from signalforge.integrity import verify_freeze,authorize_final,forward_clock


def test_atomic_commit_corruption(tmp_path):
    commit_bundle(tmp_path,{'metrics.json':{'n':1}},{'id':'x'})
    validate_bundle(tmp_path)
    (tmp_path/'metrics.json').write_text('{}')
    with pytest.raises(ValueError):validate_bundle(tmp_path)


def test_immutable_identity(tmp_path):
    commit_bundle(tmp_path,{'x.json':[1]},{'id':'a'})
    with pytest.raises(ValueError):commit_bundle(tmp_path,{'x.json':[2]},{'id':'b'})


def test_no_nan_receipt(tmp_path):
    with pytest.raises(ValueError):atomic_json(tmp_path/'x',{'x':float('nan')})
    assert not (tmp_path/'x').exists()


def test_admission_host_required():
    g=1024**3
    assert not admission(g,g,g,20*g,14*g,16*g,100*g,None)
    assert admission(g,g,g,20*g,14*g,16*g,100*g,100*g)
    assert not admission(9*g,g,g,20*g,14*g,16*g,100*g,100*g)


@pytest.mark.parametrize('url',['https://sec.gov.evil.org/a','http://www.eia.gov/a','https://sec.gov@evil.org/a','https://sec.gov:8443/a'])
def test_allowlist_reject(url):assert not allowed_url(url)


def test_allowlist_and_redaction():
    assert allowed_url('https://api.stlouisfed.org/fred/series/observations')
    assert 'secret' not in redact_url('https://api.stlouisfed.org/x?api_key=secret')


@pytest.mark.parametrize('name',['../escape','/root','x:y','a\\b'])
def test_zip_paths(tmp_path,name):
    p=tmp_path/'x.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(name,'a,b\n1,2')
    with pytest.raises(ValueError):validate_payload(p,'zip')


def test_zip_crc_and_budget(tmp_path):
    p=tmp_path/'x.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('a.csv','a,b\n1,2')
    assert validate_payload(p,'zip')['crc_verified']
    with pytest.raises(ValueError):validate_payload(p,'zip',max_uncompressed=2)


def test_html_denial(tmp_path):
    p=tmp_path/'x.csv';p.write_text('<!DOCTYPE html>denied')
    with pytest.raises(ValueError):validate_payload(p,'csv')


def test_discover_no_constructed_url():
    h='<a href="/files/dea/history/fut_fin_txt_2022.zip">text</a>'
    assert discover_zip(h,'cftc',2022).endswith('2022.zip')
    with pytest.raises(ValueError):discover_zip(h,'cftc',2021)


def test_nport_reference_not_zip_quarter():
    assert nport_reference_months('2022-01-31')==['2021-11','2021-12','2022-01']


def test_train_only_transform():
    x=np.array([[1,np.nan,2],[3,np.nan,2]])
    t=TrainTransform().fit(x,['2020-01-01','2020-01-02'],'2020-01-03T00:00Z')
    assert np.isfinite(t.transform(x)).all()
    original=t.manifest()
    t.transform([[100,100,100]])
    assert t.manifest()==original
    with pytest.raises(ValueError):TrainTransform().fit(x,['2020-01-01','2021-01-01'],'2020-01-03T00:00Z')


def test_shared_normalizer():
    s,i=shared_target_scale([1,2,3],['2020-01-01']*3,'2020-01-02T00:00Z')
    s2,i2=shared_target_scale([1,2,3],['2020-01-01']*3,'2020-01-02T00:00Z')
    assert (s,i)==(s2,i2)
    with pytest.raises(ValueError):shared_target_scale([1],['2021-01-01'],'2020-01-02T00:00Z')


def test_actual_interval_purge():
    f=pd.DataFrame({'decision_time':['2020-01-01T00:00Z'],'label_start':['2020-01-02T00:00Z'],
                    'label_end':['2020-01-09T00:00Z'],'label_available_at':['2020-01-10T00:00Z']})
    assert len(purged_training(f,'2020-01-11T00:00Z',[('2020-01-08T00:00Z','2020-01-12T00:00Z')]))==0


def test_outage_descendants():
    assert dependency_closure({'net':['cftc'],'interaction':['net','vol'],'vol':['market']},{'cftc'})=={'cftc','net','interaction'}


@pytest.mark.parametrize('family',['historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'])
def test_statistical_save_parity(tmp_path,family):
    x=np.arange(30).reshape(-1,1)/30;y=x[:,0]**2
    m=Statistical(family,.01).fit(x,y)
    mean,q=m.predict(x)
    assert q.shape==(30,5) and (np.diff(q,axis=1)>=0).all()
    m.save(tmp_path/'model');loaded=m.load(tmp_path/'model')
    np.testing.assert_array_equal(loaded.predict(x)[0],mean)
    assert m.device=='cpu'


def test_prequential_no_in_sample():
    dates=np.arange(60);x=dates[:,None]/60;y=x[:,0]
    m,q=prequential_base(x,y,dates,10,5,label_available_at=dates+.5,label_end=dates+.5)
    assert np.isnan(m[:10]).all() and np.isfinite(m[10:]).all()
    y2=y.copy();y2[50:]=1000
    m2,_=prequential_base(x,y2,dates,10,5,label_available_at=dates+.5,label_end=dates+.5)
    np.testing.assert_array_equal(m[:50],m2[:50])


def test_missing_predictions_grid_preserved():
    grid=pd.DataFrame({'decision_time':['2020-01-01','2020-01-08'],'asset':['x','x'],'y':[0,4]})
    fallback=grid[['decision_time','asset']].assign(q0_5=0).rename(columns={'q0_5':'q0.5'})
    pred=fallback.iloc[:1]
    out=score_grid(grid,pred,fallback,[.5],1)
    assert out['fallback_count']==1 and out['n_unique_dates']==2 and out['score']==1


def test_holm_missing_family_member():
    out=holm({'a':.01,'b':None,'c':.04})
    assert out=={'a':.03,'c':.08,'b':None}


def test_paired_grid_mismatch():
    with pytest.raises(ValueError):compare({'a':1},{'b':1})


def test_calibration_support_and_ensemble():
    dates=pd.date_range('2023-07-01',periods=26,freq='7D',tz='UTC')
    minimum={'0.05':52,'0.1':39,'0.5':20,'0.9':39,'0.95':52}
    c=Calibrator().fit(np.ones(26),np.zeros((26,5)),dates,dates,'ensemble','2023-07-01T00:00Z','2023-12-31T23:59Z',minimum)
    np.testing.assert_array_equal(c.corrections,[0,0,1,0,0])
    assert c.transform(np.zeros((1,5)),'ensemble')['crossing_rows']==1
    with pytest.raises(ValueError):c.transform(np.zeros((1,5)),'different')


def test_split_dividend_and_ex_entitlement():
    b=Ledger(100)
    b.rebalance({'x':1},{'x':10})
    b.split('x',2,'split')
    assert b.nav({'x':5})==100
    b.ex_dividend('x',1,'div')
    assert b.nav({'x':4})==100
    b.rebalance({}, {'x':4})
    b.pay_dividend('div')
    assert b.cash==100
    with pytest.raises(ValueError):b.ex_dividend('x',1,'div')


def test_new_ex_date_purchase_no_prior_entitlement():
    b=Ledger(100)
    b.ex_dividend('x',1,'div')
    b.rebalance({'x':1},{'x':10})
    b.pay_dividend('div')
    assert b.cash==0 and b.nav({'x':10})==100


def test_drifted_turnover_and_cost():
    b=Ledger(100)
    b.rebalance({'x':.5},{'x':10})
    result=b.rebalance({'x':.5},{'x':20},10)
    assert result['traded_notional']>25
    assert result['fee']==pytest.approx(result['traded_notional']*.001)
    assert result['nav_after']==pytest.approx(150-result['fee'])


def test_cash_opportunity_controller():
    w,status=feasible_weights([.001,.001],.002,np.eye(2)*.0001)
    assert w.sum()<1e-5


def test_final_denied_flags_without_evidence(tmp_path):
    auth={'authorized':True,'study_id':'sgqx-v3','scope':'one_registered_frozen_batch_after_all_track_gates'}
    with pytest.raises(PermissionError):verify_freeze({'study_id':'sgqx-v3','status':'READY_FOR_FINAL','protocol_hash':'x'},tmp_path,tmp_path,auth)
    with pytest.raises(PermissionError):authorize_final({},tmp_path,tmp_path,auth,'development')


def test_prospective_clock_replay_rejected():
    with pytest.raises(PermissionError):forward_clock('2025-01-01T00:00Z','2024-01-01T00:00Z','2024-01-08T00:00Z','2026-01-01T00:00Z')
    assert forward_clock('2026-01-01T00:00Z','2026-01-02T00:00Z','2026-01-08T00:00Z','2026-01-02T00:00Z')
