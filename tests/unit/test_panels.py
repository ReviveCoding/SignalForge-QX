import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from signalforge.panels import FeatureSpec,fixed_grid,build_features,matched_grid,track_folds


def card():return {'assets':['SPY','QQQ'],'outcome_blind':True,'frozen_before_outcomes':'2017-01-01T00:00Z'}


def events():
    return pd.DataFrame([{'entity':'US','source':'eia','field':'stocks','reference_time':reference,'available_at':available,'value':value,'unit':'million_barrels','raw_hash':raw*64,'pit_tier':'B'} for reference,available,value,raw in [('2020-01-01T00:00Z','2020-01-02T00:00Z',100.,'a'),('2020-01-08T00:00Z','2020-01-09T00:00Z',110.,'b'),('2020-01-01T00:00Z','2020-01-15T00:00Z',900.,'c')]])


def test_shared_feature_vectors_lineage_percentile_and_future_invariance():
    grid=fixed_grid(['2020-01-10T23:00Z'],card());spec=[FeatureSpec('stocks','eia','stocks','US','million_barrels',percentile_window=26)]
    frame=events();before=build_features(grid,frame.iloc[:2],spec,['eia']);after=build_features(grid,frame,spec,['eia'])
    assert np.array_equal(before.x.tolist(),after.x.tolist())
    assert before.x.iloc[0][0]==110 and before.x.iloc[0][3:6]==[1.,1.,1.]
    assert before.feature_lineage.iloc[0]['parents'][0]['raw_hash']=='b'*64
    assert len(before.attrs['feature_names'])==8
    assert matched_grid(before,after)


def test_missing_source_retains_eligible_grid_and_two_ages():
    grid=fixed_grid(['2020-01-01T23:00Z','2020-01-10T23:00Z'],card());spec=[FeatureSpec('stocks','eia','stocks','US','million_barrels')]
    panel=build_features(grid,events(),spec,['eia'])
    assert len(panel)==4 and np.isnan(panel.x.iloc[0][0])
    assert panel.x.iloc[0][3:5]==[0.,0.]
    assert panel.x.iloc[2][1]!=panel.x.iloc[2][2]
    with pytest.raises(ValueError,match='grid differs'):matched_grid(panel,panel.iloc[1:])


def test_grid_rejects_outcome_selection_naive_dates_and_reserved():
    with pytest.raises(PermissionError):fixed_grid(['2020-01-01T00:00Z'],{**card(),'outcome_blind':False})
    with pytest.raises(ValueError,match='Aware'):fixed_grid(['2020-01-01'],card())
    with pytest.raises(PermissionError,match='reserved'):fixed_grid(['2024-01-01T00:00Z'],card())


def test_nested_has_only_own_two_folds_and_mature_training():
    study=json.loads((Path(__file__).parents[2]/'configs/study.json').read_text())
    dates=pd.date_range('2019-10-04', '2022-12-30',freq='W-FRI',tz='UTC')
    frame=pd.DataFrame({'decision_time':dates.astype(str),'label_start':dates.astype(str),'label_end':(dates+pd.Timedelta(days=7)).astype(str),'label_available_at':(dates+pd.Timedelta(days=8)).astype(str),'y':np.ones(len(dates))})
    folds,blocked=track_folds(frame,study,'Nested-B')
    assert not blocked and set(folds)=={2021,2022}
    for year,fold in folds.items():
        assert pd.to_datetime(fold['train'].label_available_at,utc=True).max()<pd.Timestamp(f'{year}-01-01T00:00Z')
        assert pd.to_datetime(fold['inner'].label_available_at,utc=True).max()<pd.to_datetime(fold['valid'].decision_time,utc=True).min()


def test_context_masks_preserve_short_history_and_missing_distinctly():
    from signalforge.panels import build_contexts
    grid=fixed_grid(['2020-01-03T23:00Z','2020-01-10T23:00Z'],card())
    panel=build_features(grid,events(),[FeatureSpec('stocks','eia','stocks','US','million_barrels')],['eia'])
    frame,x,masks=build_contexts(panel,lookback=2)
    assert len(frame)==4 and x.shape==(4,2,8)
    assert masks['padding'][0].tolist()==[True,False]
    assert masks['eligible'].tolist()==[False,True,False,True]
    assert not masks['observed'][0,0].any() and masks['observed'][0,1].any()
    assert x[0,:, -1].tolist()==[1.,0.]
    assert len(frame.context_lineage.iloc[1])==2
    assert all(pd.Timestamp(r['max_dependency_available_at'])<=pd.Timestamp(r['decision_time']) for r in frame.context_lineage.iloc[1])


def test_invalid_tier_c_never_implicitly_promoted():
    frame=events();frame['pit_tier']='C';grid=fixed_grid(['2020-01-10T23:00Z'],card());spec=[FeatureSpec('stocks','eia','stocks','US','million_barrels')]
    with pytest.raises(PermissionError,match='Tier C'):build_features(grid,frame,spec,['eia'])
    panel=build_features(grid,frame,spec,['eia'],allow_invalid_control=True)
    assert not panel.attrs['promotion_eligible']
