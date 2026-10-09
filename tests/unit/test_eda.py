import pandas as pd
import pytest
from signalforge.eda import event_audit


def events():
    return pd.DataFrame([{'entity':'US','source':'eia','field':'stocks','unit':'million_barrels','raw_hash':'a'*64,'pit_tier':'B','reference_time':'2020-01-01T00:00Z','available_at':date,'value':value} for date,value in [('2020-01-02T00:00Z',0.),('2020-01-03T00:00Z',None)]])


def test_raw_qa_zero_missing_revision_and_order():
    frame=events();result=event_audit(frame)
    assert result['canonical_content_id']==event_audit(frame.iloc[::-1])['canonical_content_id']
    row=result['series'][0]
    assert row['true_zero']==1 and row['missing']==1 and row['revisions']==1
    assert row['revision_from_first_public_days']['max']==1
    assert result['outlier_deletions']==0 and not result['tier_a_qualified']


def test_raw_qa_rejects_reserved_rows():
    frame=events();frame['available_at']='2024-01-02T00:00Z'
    with pytest.raises(PermissionError,match='reserved'):event_audit(frame)


def test_true_extremes_retained_and_conflicting_units_rejected():
    frame=events();frame['value']=[-1e9,1e9]
    result=event_audit(frame)
    assert result['rows']==2 and result['outlier_deletions']==0
    assert result['series'][0]['minimum']==-1e9 and result['series'][0]['maximum']==1e9
    frame.loc[1,'unit']='barrels'
    with pytest.raises(ValueError,match='Conflicting canonical units'):event_audit(frame)
