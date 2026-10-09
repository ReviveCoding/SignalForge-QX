import numpy as np
import pandas as pd
import pytest
from signalforge.auxiliary import sequences,run
from signalforge.data import auxiliary_rows


def fixture_records(n=40):
    records=[]
    for issue in pd.date_range('2017-01-04',periods=n,freq='7D',tz='UTC'):
        reference=issue-pd.Timedelta(days=5)
        event={'available_at':(issue+pd.Timedelta(days=1)).isoformat(),'reference_time':reference.isoformat(),
               'pit_tier':'B','series':{k:{'value':400.,'change':float(i)} for i,k in enumerate(['Commercial (Excluding SPR)','Cushing','Distillate Fuel Oil','SPR','Total Motor Gasoline'])}}
        records.append({'state':'SUCCEEDED','event':event,'raw':{'sha256':str(issue)}})
    return records


def test_auxiliary_target_after_decision():
    frame=auxiliary_rows(fixture_records())
    scored=frame[frame.y.notna()]
    assert (pd.to_datetime(scored.label_end)>pd.to_datetime(scored.decision_time)).all()
    assert (pd.to_datetime(frame.max_dependency_available_at)<=pd.to_datetime(frame.decision_time)).all()
    assert len(frame)==40 and frame.y.isna().sum()==1
    rows,x=sequences(frame)
    assert len(rows)==14 and x.shape==(14,26,14)


def test_auxiliary_context_missing_week_not_hidden():
    records=fixture_records();records.pop(20)
    frame=auxiliary_rows(records)
    rows,x=sequences(frame)
    # Missing target is excluded, but its known feature context is retained.
    assert len(rows)==14 and frame.y.isna().sum()==2


def test_auxiliary_incomplete_panel_gate(tmp_path):
    import json
    p=tmp_path/'reports';p.mkdir()
    (p/'eia_development_acquisition.json').write_text(json.dumps({'completed':1,'required':3}))
    with pytest.raises(RuntimeError,match='complete discovered chronological'):run(tmp_path,tmp_path)


def test_holiday_friday_release_preserves_weekly_asof_grid():
    records=fixture_records()
    issue=records[20]['event']
    issue['issue_date']='2017-05-26'
    issue['available_at']='2017-05-27T04:00:00Z'
    frame=auxiliary_rows(records)
    origin=frame[frame.decision_time.eq('2017-05-26T22:00:00+00:00')].iloc[0]
    assert origin.raw_hash==records[19]['raw']['sha256']
    assert origin.target_raw_hash==records[21]['raw']['sha256']
    assert pd.Timestamp(origin.max_dependency_available_at)<=pd.Timestamp(origin.decision_time)
    rows,x=sequences(frame)
    assert len(rows)==14 and len(frame)==40
