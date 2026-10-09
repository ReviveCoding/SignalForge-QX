import numpy as np,pandas as pd,pytest
from diagnostics_v3.core import *

def panel():
    return pd.DataFrame([{'decision_time':f'2022-01-{d:02d}T22:00:00Z','asset':a,'seed':s,'target':0.,'scale':1.,'q05':-2.,'q10':-1.,'q50':0.,'q90':1.,'q95':2.} for d in (7,14) for a in ('A','B') for s in (11,37,71)])

def test_complete_grid():assert check_grid(panel(),dates=2,assets=2)
def test_partial_date_fails():
    with pytest.raises(ValueError):check_grid(panel().iloc[:-1],dates=2,assets=2)
def test_duplicate_identity_fails():
    f=panel();f.iloc[0]=f.iloc[1]
    with pytest.raises(ValueError):check_grid(f,dates=2,assets=2)
def test_crossing_rejected():
    f=panel();f.loc[0,'q05']=5
    with pytest.raises(ValueError):check_grid(f,dates=2,assets=2)
def test_nonfinite_rejected():
    f=panel();f.loc[0,'q50']=np.nan
    with pytest.raises(ValueError):check_grid(f,dates=2,assets=2)
def test_reserved_rejected():
    f=panel();f.decision_time='2024-01-05T22:00:00Z'
    with pytest.raises(PermissionError):check_grid(f,dates=2,assets=2)
def test_seed_scale_parity():
    f=panel();f.loc[0,'scale']=2
    with pytest.raises(ValueError):check_grid(f,dates=2,assets=2)
def test_exact_pinball():
    f=augment(panel());assert np.allclose(f.loss,.08);assert summary(f)['coverage90']==1.
def test_outer_threshold_forbidden():
    with pytest.raises(PermissionError):train_only_threshold([1],['2022-01-07Z'],['2022-01-14Z'],'2023-01-01T00:00Z',cohort='OUTER')
def test_maturity_threshold_forbidden():
    with pytest.raises(PermissionError):train_only_threshold([1],['2021-01-01T00:00Z'],['2023-01-01T00:00Z'],'2022-01-01T00:00Z',cohort='MATURE_PRE_OUTER')
def test_train_only_threshold():
    x=train_only_threshold([1,2],['2021-01-01T00:00Z','2021-01-08T00:00Z'],['2021-01-03T00:00Z','2021-01-10T00:00Z'],'2022-01-01T00:00Z',cohort='MATURE_PRE_OUTER',minimum_dates=2);assert x['thresholds']['0.95']==1.95 and not x['outer_outcomes_used']
def test_small_cohort_underpowered():
    x=train_only_threshold([1],['2021-01-01T00:00Z'],['2021-01-03T00:00Z'],'2022-01-01T00:00Z',cohort='MATURE_PRE_OUTER');assert x['state']=='UNDERPOWERED'
def test_no_model_oof_calibration():assert model_error_threshold('SIA_v2',False)['state']=='BLOCKED_OOF_UNAVAILABLE'
def test_p1_claim_rejected():
    with pytest.raises(PermissionError):economic_claim()
def test_weights_caps():
    for m in ['top1','top2','linear_rank']:
        a=['GLD','SLV','UNG','USO','SPY','QQQ','IEF','TLT'];w=capped_weights(a,np.arange(8)[::-1],m);assert w.max()<=.25 and w[:4].sum()<=.35+1e-12 and w.sum()<=1

def test_rank_ties_and_switch():
    a=panel().query('seed==11').iloc[:2];b=a.copy();b['q50']=[1,-1];x=rank_compare(a,b);assert x['spearman'] is None and x['ties_a']==1

def test_bootstrap_independent_dates():
    f=augment(panel());x=bootstrap_dates(f,['loss'],blocks=(1,),draws=100);assert x['1']['independent_dates']==2 and np.allclose(x['1']['columns']['loss']['interval95'],[.08,.08])
def test_realized_slices_do_not_change_scores():
    f=augment(panel());f['outcome_slice']=tercile(f.target,[-.5,.5]);assert summary(f)['normalized_pinball']==pytest.approx(.08)
def test_json_tuple_keys_normalized_without_loosening_identity():
    assert canonical_keys([('2022-01-07T22:00:00Z','SPY')])==canonical_keys([['2022-01-07 22:00:00+00:00','SPY']])
    assert canonical_keys([('2022-01-07T22:00:00Z','SPY')])!=canonical_keys([['2022-01-07T22:00:00Z','QQQ']])
def test_naive_decision_key_rejected():
    with pytest.raises(ValueError):canonical_keys([('2022-01-07','SPY')])

def test_backend_discrepancy_not_silently_passed():
    assert not replay_status(np.ones((2,5))*1.0001,np.ones((2,5)))['strict_replay_passed']
    assert replay_status(np.ones((2,5)),np.ones((2,5)))['strict_replay_passed']
def test_replay_corruption_rejected():
    with pytest.raises(ValueError):replay_status(np.array([[np.nan]]),np.ones((1,1)))

def test_panel_metadata_is_not_inference_local_state():
    m=panel_bundle_metadata('a'*64);assert m['training'] is False and m['reserved_access'] is False and 'replay_status' not in m
