import numpy as np,pandas as pd,pytest
from calibration_v34.api import *
from signalforge.v33_sia import Feature,SemanticAgeTransform
from signalforge.v33_corrections import mature_partitions

def data(n=60):
    d=pd.date_range('2018-01-05',periods=n,freq='7D',tz='UTC');return pd.DataFrame({'decision_time':d,'label_end':d+pd.Timedelta(days=1),'label_available_at':d+pd.Timedelta(days=2),'target':np.arange(n)/n,'scale':1.,'q05':-.5,'q10':-.4,'q50':0.,'q90':.4,'q95':.5,'role':'OOF'})
def test_mature_actual_dates():assert maturity(data(),'2020-01-01T00:00Z')==60
def test_outer_calibration_forbidden():
    f=data();f.role='OUTER'
    with pytest.raises(PermissionError):fit(f,'intercept','2020-01-01T00:00Z')
def test_immature_label_trap():
    f=data();f.loc[0,'label_available_at']='2021-01-01T00:00Z'
    with pytest.raises(PermissionError):maturity(f,'2020-01-01T00:00Z')
def test_end_after_cutoff_trap():
    f=data();f.loc[0,'label_end']='2021-01-01T00:00Z'
    with pytest.raises(PermissionError):maturity(f,'2020-01-01T00:00Z')
def test_reserved_cohort_rejected():
    f=data();f.loc[0,'decision_time']='2024-01-01T00:00Z'
    with pytest.raises(PermissionError):maturity(f,'2025-01-01T00:00Z')
def test_pooled_assets_do_not_multiply_dates():assert maturity(pd.concat([data()]*8),'2020-01-01T00:00Z')==60
def test_sparse_tails_identity():
    p=fit(data(),'intercept','2020-01-01T00:00Z');assert p['supported']==[False,False,True,False,False];z=apply(data()[C],data().scale,p);assert np.array_equal(z[:,[0,1,3,4]],data()[C].to_numpy()[:,[0,1,3,4]])
def test_mature_main_tails_support():
    f=data(428)
    for column in ('decision_time','label_end','label_available_at'):
        f[column]=f[column]-pd.Timedelta(days=2920)
    assert all(fit(f,'intercept','2020-01-01T00:00Z')['supported'])
def test_identity_exact():assert np.array_equal(apply(data()[C],data().scale,fit(data(),'identity','2020-01-01T00:00Z')),data()[C])
def test_noncrossing_fixed_unsupported_anchors():
    p=fit(data(),'intercept','2020-01-01T00:00Z');p['offset_normalized'][2]=100;z=apply(data()[C],data().scale,p);assert (np.diff(z,axis=1)>=0).all() and np.array_equal(z[:,4],data().q95)
def test_weighted_quantile_equal_date():assert wquantile([1,2,3],.5,[1,1,1])==2
def test_clock_off_by_one():
    with pytest.raises(PermissionError):monitor(np.array([[-2,-1,0,1,2]]),[1],{},known_at='2022-01-01T00:00:01Z',decision_time='2022-01-01T00:00:00Z')
def test_monitor_not_live():assert monitor(np.array([[-2,-1,0,1,2]]),[1],{'width90_p99':3},known_at='2022-01-01T00:00Z',decision_time='2022-01-01T00:00Z')['live_notification_sent'] is False
def test_feature_future_fit_rejected():
    with pytest.raises(PermissionError):SemanticAgeTransform([Feature('v','numeric','market')]).fit([[1]],['v'],['2020-01-01T00:00Z'],'2020-01-01T00:00Z',available_at=['2020-01-01T00:00:01Z'])
def test_prefix_scale_no_future_mutation():
    s=SemanticAgeTransform([Feature('v','numeric','market')]).fit([[1],[2]],['v'],['2019-01-01T00:00Z','2019-01-02T00:00Z'],'2019-01-03T00:00Z',available_at=['2019-01-01T00:00Z','2019-01-02T00:00Z']);h=s.manifest();s.transform([[1e9]],['v']);assert s.manifest()==h
def test_prequential_maturity_disjoint():
    f=data();p=mature_partitions(f.decision_time,f.label_end,f.label_available_at,minimum=26,block=13)
    for block in p:assert set(block['train']).isdisjoint(block['oof']) and pd.to_datetime(f.iloc[block['train']].label_available_at,utc=True).max()<=pd.Timestamp(block['cutoff'])
def test_no_extra_candidates():
    with pytest.raises(ValueError):fit(data(),'HPO','2020-01-01T00:00Z')
def test_crossings_rejected():
    with pytest.raises(ValueError):apply(np.array([[1,0,2,3,4]]),[1],{'offset_normalized':[0]*5,'supported':[False]*5})