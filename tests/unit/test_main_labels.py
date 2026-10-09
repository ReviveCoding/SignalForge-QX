import numpy as np
import pandas as pd
import pytest
from signalforge.main_labels import build_main_labels


def marks():
    return pd.DataFrame([{'asset':asset,'time':time,'available_at':pd.Timestamp(time)+pd.Timedelta(minutes=1),'open':price,'fresh':True,'adjusted':False,'raw_hash':'a'*64}
        for asset in ['SPY','QQQ'] for time,price in [('2020-01-06T14:30Z',100.),('2020-01-13T14:30Z',110.)]])


def test_p1_fixed_grid_actual_maturity_and_missing_mark_not_dropped():
    grid=pd.DataFrame([{'decision_time':'2020-01-03T23:00Z','asset':asset,'eligible':True} for asset in ['SPY','QQQ']])
    prices=marks();prices=prices[~(prices.asset.eq('QQQ')&prices.time.eq('2020-01-13T14:30Z'))]
    coverage=[{'asset':asset,'start':'2020-01-01T00:00Z','end':'2020-02-01T00:00Z','complete':True,'raw_hash':'b'*64} for asset in ['SPY','QQQ']]
    result=build_main_labels(grid,prices,[],action_coverage=coverage)
    assert len(result)==2 and result.eligible.all() and result.y.iloc[0]==pytest.approx(.1)
    assert np.isnan(result.y.iloc[1]) and result.label_available_at.iloc[1] is None
    assert result.label_available_at.iloc[0]=='2020-01-13T14:31:00+00:00' and not result.attrs['economic_qualified']
    missing=build_main_labels(grid,prices,[])
    assert missing.target_status.eq('BLOCKED_ACTION_LEDGER_COVERAGE').all() and missing.y.isna().all()


def test_four_week_actual_calendar_and_reserved_target_boundary():
    grid=pd.DataFrame([{'decision_time':'2020-03-06T23:00Z','asset':'SPY','eligible':True}])
    result=build_main_labels(grid,marks(),[],horizon_rebalances=4)
    assert result.label_start.iloc[0]=='2020-03-09T13:30:00+00:00' and result.label_end.iloc[0]=='2020-04-06T13:30:00+00:00'
    grid['decision_time']='2023-12-29T23:00Z';result=build_main_labels(grid,marks(),[])
    assert result.target_status.iloc[0]=='OUTSIDE_DEVELOPMENT_LABEL_BOUNDARY' and np.isnan(result.y.iloc[0])
    grid['decision_time']='2024-01-05T23:00Z'
    with pytest.raises(PermissionError,match='reserved'):build_main_labels(grid,None,[])


def test_no_p0_silent_main_target_switch():
    with pytest.raises(PermissionError,match='separate'):build_main_labels(None,None,[],price_mode='P0')
