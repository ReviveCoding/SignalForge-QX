import numpy as np
import pandas as pd
import pytest
from signalforge.features import TrainTransform
from signalforge.forensics import forecast_rows,extrapolation,date_score,relative_gain,feature_lineage_ids


def test_actual_constant_training_age_can_amplify_finite_test_age_without_target_rescaling():
    transform=TrainTransform().fit(np.full((3,1),2.),['2020-01-01T00:00Z']*3,'2020-02-01T00:00Z')
    row,summary=extrapolation(np.array([[[9.]]]),transform.manifest(),['release_age'])
    assert row['transformed_abs_max'][0]==7e8
    assert summary[0]['scale_floor'] and summary[0]['test_max']==9


def test_absolute_and_normalized_pinball_and_equal_date_scoring_are_distinct():
    frame=pd.DataFrame({'decision_time':['2021-01-01T00:00Z']*2+['2021-01-08T00:00Z'],'asset':['A','B','A'],'y':[1.,2.,4.],'eligible':[True]*3})
    rows=forecast_rows(frame,np.zeros(3),np.zeros((3,5)),np.array([2.,2.,2.]))
    assert np.allclose(rows.absolute_pinball,[.5,1,2])
    assert date_score(rows)==pytest.approx((.375+1)/2)
    assert relative_gain(0.,1.) is None


def test_reserved_boundary_rejected_before_numeric_target_conversion():
    frame=pd.DataFrame({'decision_time':['2024-01-01T00:00Z'],'asset':['A'],'y':['secret_not_read'],'eligible':[True]})
    with pytest.raises(PermissionError):forecast_rows(frame,[0],np.zeros((1,5)),[1])


def test_original_crossing_and_invalid_scale_fail_closed():
    frame=pd.DataFrame({'decision_time':['2021-01-01T00:00Z'],'asset':['A'],'y':[1.],'eligible':[True]})
    with pytest.raises(ValueError):forecast_rows(frame,[0],[[1,0,0,0,0]],[1])
    with pytest.raises(ValueError):forecast_rows(frame,[0],np.zeros((1,5)),[0])


def test_actual_panel_provenance_is_per_feature_without_top_level_raw_hash():
    frame=pd.DataFrame({'feature_lineage':[{'parents':[{'raw_hash':'abc','feature':'return'}]}]})
    assert len(feature_lineage_ids(frame)[0])==64
    with pytest.raises(ValueError):feature_lineage_ids(pd.DataFrame({'raw_hash':['abc']}))

def test_actual_price_card_records_index_without_dataframe_assumption():
    from signalforge.forensics import price_mark_index
    rows=[{'asset':'USO','time':'2021-01-01T21:00:00Z','close':10,'raw_hash':'abc'}]
    assert price_mark_index(rows)[('USO','2021-01-01 21:00:00+00:00')].close==10
    with pytest.raises(ValueError):price_mark_index(rows+rows)
    with pytest.raises(PermissionError):price_mark_index([{**rows[0],'time':'2024-01-01T21:00Z','close':'unreadable reserved'}])
