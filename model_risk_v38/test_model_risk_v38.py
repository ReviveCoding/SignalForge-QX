"""TEST_ONLY fixtures; no public-data performance claims."""
import numpy as np,pandas as pd,pytest
from .source import validate_predictions,guard_path,aware,MODELS,QC,KEY,source_hash,FROZEN
from .diagnostics import enrich,mean_equal,block_bounds,paired,calibration
from .monitor_replay import mature_rows,require_mature,threshold_for,event_guard
from .portfolio_proxy import weights,validate_weights

@pytest.fixture
def panel():
 rows=[]
 for t,y in [('Main-A',2020),('Nested-B',2022)]:
  for date in pd.date_range(f'{y}-01-03',periods=52,freq='7D',tz='UTC'):
   for a in ['SPY','QQQ','IEF','TLT','GLD','SLV','UNG','USO']:
    for s in [11,37,71]:
     for m in MODELS:rows.append({'track':t,'decision_time':date,'asset':a,'seed':s,'model':m,'target':0.,'scale':1.,'q05':-2.,'q10':-1.,'q50':0.,'q90':1.,'q95':2.,'fold_id':t,'source_id':t,'target_id':t,'normalizer_id':t})
 return pd.DataFrame(rows)

def test_full_alignment(panel):assert len(validate_predictions(panel))==12480
@pytest.mark.parametrize('column,value',[('target',np.nan),('scale',0),('scale',-1),('q50',np.inf),('q05',3)])
def test_numeric_injections(panel,column,value):
 panel.loc[0,column]=value
 with pytest.raises(ValueError):validate_predictions(panel)
def test_duplicate_row(panel):
 panel.iloc[0]=panel.iloc[1]
 with pytest.raises(ValueError):validate_predictions(panel)
def test_missing_rows(panel):
 with pytest.raises(ValueError):validate_predictions(panel.iloc[:-1])
def test_missing_model(panel):
 with pytest.raises(ValueError):validate_predictions(panel[panel.model!='BAR'])
def test_wrong_track(panel):
 panel.loc[0,'track']='Unknown'
 with pytest.raises(ValueError):validate_predictions(panel)
def test_reserved(panel):
 panel.loc[0,'decision_time']=pd.Timestamp('2024-01-01T00:00Z')
 with pytest.raises(PermissionError):validate_predictions(panel)
def test_naive_timezone():
 with pytest.raises(ValueError):aware(['2020-01-01'])
def test_timezone_equivalence():assert aware(['2019-12-31T19:00:00-05:00'])[0]==pd.Timestamp('2020-01-01T00:00:00Z')
@pytest.mark.parametrize('column,value',[('target',1),('scale',2),('source_id','wrong'),('normalizer_id','wrong'),('fold_id','wrong'),('target_id','wrong')])
def test_exact_identity_mismatch(panel,column,value):
 panel.loc[0,column]=value
 with pytest.raises(ValueError):validate_predictions(panel)
def test_exact_pinball(panel):
 x=enrich(panel);assert x.loss.iloc[0]==pytest.approx(.08);assert x.raw_loss.iloc[0]==pytest.approx(.08);assert x.coverage90.mean()==1
@pytest.mark.parametrize('scale',[.1,2.,10.])
def test_normalization(panel,scale):
 panel['scale']=scale;x=enrich(panel);assert x.loss.iloc[0]==pytest.approx(.08/scale)
def test_interval_score_outside(panel):
 panel['target']=3;x=enrich(panel);assert x.interval_score90.iloc[0]==24;assert x.interval_score80.iloc[0]==22

def test_signed_bias(panel):
 panel['target']=-3;x=enrich(panel);assert x.normalized_bias.iloc[0]==-3

def test_date_equal_not_row_equal():
 p=pd.DataFrame({'decision_time':[1,2,2,2],'asset':['A','A','B','B'],'seed':[11,11,11,37],'x':[10,0,0,0]});assert mean_equal(p,'x')==5

def test_repeated_seeds_not_dates(panel):
 a=calibration(enrich(panel[(panel.track=='Main-A')&(panel.model=='SIA_v2')]));assert a['distinct_dates']==52 and a['rows']==1248

def test_stress_preserved(panel):
 panel.loc[(panel.asset=='USO'),'target']=100;v=validate_predictions(panel);assert len(v)==12480;assert enrich(v).loss.max()>10

def test_pair_zero(panel):assert paired(enrich(panel),'SIA_v2','strong_I0').delta.max()==0

def test_pair_missing(panel):
 p=enrich(panel);p=p.drop(p.index[(p.model=='BAR')][0])
 with pytest.raises(ValueError):paired(p,'BAR','strong_I0')
@pytest.mark.parametrize('block',[4,8,13])
def test_block_constant(block):assert block_bounds(np.ones(52),block)==(1.,1.)
def test_block_deterministic():assert block_bounds(np.arange(52),4)==block_bounds(np.arange(52),4)
def test_monitor_insufficient():assert threshold_for(np.arange(37),26,13) is None

def test_monitor_exact():
 h=np.arange(38);rolling=np.array([np.mean(h[i:i+13]) for i in range(26)]);assert threshold_for(h,26,13)==pytest.approx(np.quantile(rolling,.95))
def test_monitor_ignores_distant_history():assert threshold_for(np.r_[99999,np.arange(38)],26,13)==threshold_for(np.arange(38),26,13)
def test_monitor_rejects_nonfinite():
 with pytest.raises(ValueError):threshold_for([np.nan]*100,26,13)
def test_immature_label_rejected():
 with pytest.raises(ValueError):require_mature(['2020-01-10T00:00Z'],'2020-01-09T00:00Z')
def test_label_equal_cutoff_allowed():require_mature(['2020-01-10T00:00Z'],'2020-01-10T00:00Z')
def test_maturity_timezone_required():
 with pytest.raises(ValueError):require_mature(['2020-01-10T00:00Z'],'2020-01-10')
def test_prior_only_mature_selection():
 p=pd.DataFrame({'decision_time':pd.to_datetime(['2020-01-01Z'.replace('01Z','01T00:00Z'),'2020-01-02T00:00Z','2020-01-03T00:00Z']), 'label_available_at':pd.to_datetime(['2020-01-05T00:00Z','2020-01-09T00:00Z','2020-01-04T00:00Z'])});assert len(mature_rows(p,'2020-01-06T00:00Z','2020-01-03T00:00Z'))==1
@pytest.mark.parametrize('args',[{'pred':[1,0,2,3,4]},{'pred':[0,1,2,3,np.inf]},{'pred':[0,1,2,3,4],'feature_at':'2020-01-02T00:00Z','decision':'2020-01-01T00:00Z'},{'pred':[0,1,2,3,4],'age':-1},{'pred':[0,1,2,3,4],'age':99,'max_age':10},{'pred':[0,1,2,3,4],'age':np.nan},{'pred':[0,1,2,3,4],'expected_hash':'a','actual_hash':'b'},{'pred':[0,1,2,3,4],'checkpoint_ok':False},{'pred':[0,1,2,3,4],'available_at':'2020-01-10T00:00Z','at':'2020-01-01T00:00Z'}])
def test_hard_integrity_injections(args):
 with pytest.raises(ValueError):event_guard(**args)
def test_absent_independent_signer():
 with pytest.raises(PermissionError):event_guard([0,1,2,3,4],deployment=True)
@pytest.mark.parametrize('mapping',['top1','top2','linear_rank'])
def test_fixed_weight_caps(mapping):
 a=['GLD','SLV','UNG','USO','SPY','QQQ','IEF','TLT'];w=weights(a,np.arange(8)[::-1],mapping);assert validate_weights(a,w);assert w[:4].sum()<=.35+1e-12 and w.max()<=.25

def test_ties_lexical():assert np.array_equal(weights(['B','A'],[1,1],'top1'),[0,.25])
def test_wrong_mapping():
 with pytest.raises(ValueError):weights(['A'],[1],'optimized')
def test_overweight_guard():
 with pytest.raises(ValueError):validate_weights(['A'],[.3])
@pytest.mark.parametrize('path',['/tmp/canonical_2024.csv','/mnt/c/Users/USERNAME/Downloads/SignalForge-QX/runtime/ledger/development_compute.sqlite','/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev/reports/model_risk_v3/reserved.csv'])
def test_protected_paths(path):
 with pytest.raises(PermissionError):guard_path(path)
def test_frozen_source_hash():assert source_hash()==FROZEN
