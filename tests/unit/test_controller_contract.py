import numpy as np
import pytest
from signalforge.portfolio import Ledger,feasible_weights,controller_horizon


def test_controller_mean_head_horizon_and_asof_cash():
    forecast={'decision_time':'2020-01-03T23:00Z','horizon_weeks':1,'mean':[.03,.01],'q50':[-.4,-.5]}
    risk={'decision_time':forecast['decision_time'],'horizon_weeks':1,'covariance':[[.0001,0],[0,.0001]],'max_dependency_available_at':'2020-01-03T22:00Z'}
    cash={'decision_time':forecast['decision_time'],'horizon_weeks':1,'return':.001,'max_dependency_available_at':'2020-01-03T22:00Z'}
    weights,state=controller_horizon(forecast,risk,cash)
    assert weights.sum()>0 and state=='QUALIFIED_SOLUTION'
    with pytest.raises(ValueError,match='mismatch'):controller_horizon(forecast,{**risk,'horizon_weeks':4},cash)
    with pytest.raises(ValueError,match='future'):controller_horizon(forecast,risk,{**cash,'max_dependency_available_at':'2020-01-04T00:00Z'})


def test_locked_receivables_do_not_finance_new_exposure():
    book=Ledger(10,{'x':1},{'unpaid':10})
    weights,state=feasible_weights([.03],.001,[[.0001]],investable_fraction=.2)
    assert weights.sum()<=.2+1e-7
    with pytest.raises(ValueError):Ledger(100,{'x':-1})
    assert isinstance(book.cash,float)
