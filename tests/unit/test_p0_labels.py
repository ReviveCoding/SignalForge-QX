import numpy as np
import pandas as pd
import pytest
from signalforge.panels import fixed_grid
from signalforge.p0_labels import build_p0_labels,last_regular_close
from signalforge.track_engine import validate_panel


def test_p0_holiday_dst_known_close_and_no_economics_or_adjustment_switch():
    dates=pd.date_range('2022-04-08','2022-04-29',freq='W-FRI',tz='America/New_York')+pd.Timedelta(hours=18)
    grid=fixed_grid(dates,{'assets':['a'],'outcome_blind':True,'frozen_before_outcomes':'documented fixture'})
    marks=pd.DataFrame([{'asset':'a','decision_time':d.isoformat(),'time':last_regular_close(d).isoformat(),
        'available_at':(last_regular_close(d)+pd.Timedelta(minutes=1)).isoformat(),'close':100+i,'fresh':True,
        'adjusted':False,'raw_hash':'a'*64} for i,d in enumerate(dates)])
    labels=build_p0_labels(grid,marks,'unadjusted_close_price_return')
    assert labels.y.notna().sum()==3 and not labels.economic_qualified.any()
    assert labels.iloc[0].y==pytest.approx(.01)
    # Good Friday is closed: the actual Thursday close remains explicit.
    assert pd.Timestamp(labels.iloc[0].label_end).tz_convert('America/New_York').day==14
    labels['context_eligible']=True;labels['sequence_index']=np.arange(len(labels));labels['max_dependency_available_at']=labels.decision_time
    validate_panel(labels,np.ones((len(labels),2,2)))
    changed=labels.copy();changed.loc[0,'p0_origin_mark_available_at']='2022-04-09T00:00Z'
    with pytest.raises(ValueError,match='known'):validate_panel(changed,np.ones((len(labels),2,2)))
    with pytest.raises(ValueError,match='basis switch'):build_p0_labels(grid,marks,'vendor_adjusted_close_benchmark')
    marks.loc[1,'fresh']=False
    missing=build_p0_labels(grid,marks,'unadjusted_close_price_return')
    assert len(missing)==4 and missing.y.notna().sum()==1


def test_p0_reserved_price_clock_denied_before_numerical_read():
    grid=fixed_grid(['2023-12-22T23:00Z'],{'assets':['a'],'outcome_blind':True,'frozen_before_outcomes':'fixture'})
    marks=pd.DataFrame([{'asset':'a','decision_time':'2024-01-05T23:00Z','time':'2024-01-05T21:00Z',
        'available_at':'2024-01-05T21:01Z','close':'DO NOT INSPECT','fresh':True,'adjusted':False,'raw_hash':'a'*64}])
    with pytest.raises(PermissionError,match='Reserved'):build_p0_labels(grid,marks,'unadjusted_close_price_return')
