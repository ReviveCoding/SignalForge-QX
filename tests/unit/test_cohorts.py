import pandas as pd
import pytest
from signalforge.cohorts import cohort_audit


def test_coverage_cohort_exit_missing_and_late_amendment_are_visible():
    rows=[]
    for month,entities in [('2021-01',['A','B']),('2021-02',['A','C'])]:
        for entity in entities:
            rows.append({'reference_month':month,'series_id':entity,'available_at':'2021-03-01T00:00Z',
                'external_flow':0. if entity=='A' else 10.,'reinvestment':1.,'accession':month+entity,'reported_end_net_assets':100.,'report_date':str(pd.Period(month).end_time.date())})
    rows.append({**rows[-1],'available_at':'2021-04-01T00:00Z','external_flow':1000.,'accession':'late-amendment'})
    result=cohort_audit(pd.DataFrame(rows),['2021-01','2021-02'],'2021-03-15T00:00Z',['A','B'])
    first,last=result['months']
    assert first['fixed_external_flow']==10 and first['fixed_coverage']==1
    assert last['fixed_external_flow'] is None and last['fixed_state']=='BLOCKED_DATA'
    assert last['fixed_missing_members']==['B'] and last['observed_external_flow']==10
    assert last['entered_since_previous_observed_month']==['C'] and last['exited_since_previous_observed_month']==['B']
    assert last['true_zero_flow_entities']==1 and last['known_revisions']==0 and last['reported_assets_HHI']==.5
    rows[-1]['available_at']='2021-03-10T00:00Z'
    changed=cohort_audit(pd.DataFrame(rows),['2021-01','2021-02'],'2021-03-15T00:00Z',['A','B'])['months'][-1]
    assert changed['known_revisions']==1 and changed['observed_external_flow']==1000 and changed['fixed_external_flow'] is None
    with pytest.raises(ValueError):cohort_audit(pd.DataFrame(rows),['2021-02','2021-01'],'2021-03-15T00:00Z',['A','B'])
    with pytest.raises(PermissionError):cohort_audit(pd.DataFrame(rows),['2024-01'],'2024-03-15T00:00Z',['A','B'])
    for row in rows:row['report_date']='2021-03-31'
    assert all(card['reported_assets_HHI'] is None for card in cohort_audit(pd.DataFrame(rows),['2021-01','2021-02'],'2021-03-15T00:00Z',['A','B'])['months'])
