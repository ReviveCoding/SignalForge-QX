from tests.unit.test_auxiliary_protocol import fixture_records
from signalforge.data import auxiliary_rows
import pandas as pd


def test_same_friday_issue_is_not_future_target_from_date_upper_bound():
    records=fixture_records()
    for i,record in enumerate(records):
        issue=pd.Timestamp('2017-01-06T00:00Z')+pd.Timedelta(weeks=i)
        record['event']['issue_date']=str(issue.date())
        record['event']['reference_time']=(issue-pd.Timedelta(days=7)).isoformat()
        record['event']['available_at']=(issue.tz_convert('America/New_York').normalize()+pd.Timedelta(days=2)).isoformat()
    frame=auxiliary_rows(records)
    assert len(frame)==40 and frame.y.notna().sum()==38
    scored=frame[frame.y.notna()]
    assert (pd.to_datetime(scored.target_issue_date).dt.date>pd.to_datetime(scored.decision_time).dt.date).all()
    assert (pd.to_datetime(frame.max_dependency_available_at,utc=True)<=pd.to_datetime(frame.decision_time,utc=True)).all()
    # Each origin uses last week's known report and predicts next week's issue,
    # never the same-Friday issue obscured by a conservative upper bound.
    first=frame.iloc[0]
    assert first.raw_hash==records[0]['raw']['sha256']
    assert first.target_raw_hash==records[2]['raw']['sha256']
