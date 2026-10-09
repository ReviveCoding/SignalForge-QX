import numpy as np,pandas as pd,pytest
from .controls_current import age_candidate_guard
from .monitor_replay import event_guard,threshold_for,mature_rows
from .independent import scalar_pinball,decimal_pinball,linear_quantile

def test_train_only_age_cutoff():assert age_candidate_guard(1,'2019-12-31T00:00Z','2020-01-03T00:00Z')
@pytest.mark.parametrize('kwargs',[{'cutoff':'2020-01-03T00:00Z'},{'outer_outcomes_used':True},{'age':-1},{'cutoff':'2019-12-31'}])
def test_age_guard_injections(kwargs):
 d={'age':1,'cutoff':'2019-12-31T00:00Z','decision':'2020-01-03T00:00Z'};d.update(kwargs)
 with pytest.raises(ValueError):age_candidate_guard(**d)
@pytest.mark.parametrize('y,p,t',[(-1,2,.05),(3,-2,.95),(0,0,.5),(.25,.3,.1)])
def test_two_scalar_routines(y,p,t):assert scalar_pinball(y,p,t)==pytest.approx(float(decimal_pinball(y,p,t)))
def test_independent_quantile():assert linear_quantile([2,1,3,8],.95)==pytest.approx(np.quantile([2,1,3,8],.95))
def test_nested_66_lookback52_window26_impossible():assert threshold_for(np.ones(65),52,26) is None

def test_availability_not_decision_only():
 x=pd.DataFrame({'decision_time':pd.to_datetime(['2020-01-01T00:00Z']),'label_available_at':pd.to_datetime(['2020-02-01T00:00Z'])});assert mature_rows(x,'2020-01-20T00:00Z','2020-01-21T00:00Z').empty

def test_leading_guard_does_not_require_labels():assert event_guard([-2,-1,0,1,2],feature_at='2020-01-01T00:00Z',decision='2020-01-02T00:00Z')
