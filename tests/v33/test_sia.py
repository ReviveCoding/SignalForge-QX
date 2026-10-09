import copy
import numpy as np
import pytest
from signalforge.v33_sia import Feature, SemanticAgeTransform, schema_from_specs
from signalforge.features import TrainTransform
SCHEMA=[Feature('age','release_age','cftc'),Feature('value','numeric','cftc'),
        Feature('seen','mask','cftc'),Feature('coverage','coverage','cftc')]
NAMES=[s.name for s in SCHEMA]
def fitted(a=None):
    a=np.array([[2.750011574,4.,1.,1.],[2.750011574,4.,1.,1.]]) if a is None else a
    return SemanticAgeTransform(SCHEMA).fit(a,NAMES,['2020-01-01T00:00Z']*len(a),
        '2020-12-31T00:00Z',available_at=['2019-12-31T00:00Z']*len(a))
def test_one_day_shift_fixed_semantic_unit():
    t=fitted(); out=t.transform([[3.750011574,4.,1.,1.]],NAMES)
    assert np.isclose(out['values'][0,0],np.log1p(3.750011574))
    assert abs(np.log1p(3.750011574)-np.log1p(2.750011574))<.24
    old=TrainTransform().fit([[2.750011574],[2.750011574]],
        ['2020-01-01T00:00Z']*2,'2020-12-31T00:00Z')
    assert old.transform([[3.750011574]],False)[0,0]>9.99e7
def test_missing_is_not_zero_or_stale():
    t=fitted(); out=t.transform([[np.nan,np.nan,0.,0.],[0.,4.,1.,1.]],NAMES)
    assert out['values'][0,0]==out['values'][1,0]==0
    assert not out['observed'][0,0] and out['observed'][1,0]
    assert not out['numeric_ood'][0].any()
def test_constant_numeric_ood_separate():
    out=fitted().transform([[2.,9.,1.,1.]],NAMES)
    assert out['values'][0,1]==0 and out['numeric_ood'][0,1]
def test_unseen_zero_is_ood():
    t=fitted(np.array([[1,np.nan,0,0],[2,np.nan,0,0]]))
    assert t.transform([[1,0,1,1]],NAMES)['numeric_ood'][0,1]
@pytest.mark.parametrize('age',[-1,-1e-12,np.inf])
def test_bad_age_rejected(age):
    with pytest.raises(ValueError): fitted().transform([[age,4,1,1]],NAMES)
@pytest.mark.parametrize('mask',[.5,np.nan,2])
def test_bad_typed_mask(mask):
    with pytest.raises(ValueError): fitted().transform([[1,4,mask,1]],NAMES)
def test_no_test_fitting_and_no_future_source():
    before=fitted().manifest()
    fitted().transform([[999,10000,1,1]],NAMES)
    assert fitted().manifest()==before
    with pytest.raises(PermissionError):
        SemanticAgeTransform(SCHEMA).fit([[1,2,1,1]],NAMES,['2021-01-01T00:00Z'],
            '2020-12-31T00:00Z',available_at=['2020-01-01T00:00Z'])
    with pytest.raises(PermissionError):
        SemanticAgeTransform(SCHEMA).fit([[1,2,1,1]],NAMES,['2020-01-01T00:00Z'],
            '2020-12-31T00:00Z',available_at=['2020-01-02T00:00Z'])
def test_schema_alignment_and_duplicate():
    with pytest.raises(ValueError): fitted().transform([[1,4,1,1]],NAMES[::-1])
    with pytest.raises(ValueError): SemanticAgeTransform([SCHEMA[0],SCHEMA[0]])
    with pytest.raises(ValueError): schema_from_specs([],['release_age_trick'])
def test_manifest_roundtrip_and_corruption():
    t=fitted(); assert t.manifest()==SemanticAgeTransform.from_manifest(t.manifest()).manifest()
    m=copy.deepcopy(t.manifest());m['scale'][0]=1e-8
    with pytest.raises(PermissionError): SemanticAgeTransform.from_manifest(m)
def test_26_week_synthetic_context_boundary():
    age=np.full(53+25,2.750011574);age[25]=3.750011574
    contexts=np.array([age[i:i+26] for i in range(53)])
    assert np.array_equal(np.any(contexts!=2.750011574,axis=1),np.arange(53)<26)
    bounded=np.log1p(contexts)
    assert bounded.max()<1.56 and np.isfinite(bounded).all()
def test_ordinary_numeric_train_only_standardization():
    t=fitted(np.array([[1.,1.,1,1],[2.,3.,1,1]]))
    assert t.transform([[1.,5.,1,1]],NAMES)['values'][0,1]==3.
def test_reserved_or_naive_fit_rejected():
    with pytest.raises(PermissionError):
        SemanticAgeTransform(SCHEMA).fit([[1,2,1,1]],NAMES,['2024-01-01T00:00Z'],
            '2025-01-01T00:00Z',available_at=['2023-01-01T00:00Z'])
    with pytest.raises(ValueError):
        SemanticAgeTransform(SCHEMA).fit([[1,2,1,1]],NAMES,['2020-01-01'],
            '2021-01-01T00:00Z',available_at=['2019-01-01T00:00Z'])

def test_preparation_preserves_target_normalizer_and_source_validity():
    import pandas as pd
    from signalforge.v33_sia import prepare_sia
    from signalforge.track_engine import asset_normalizer
    dates=pd.date_range('2020-01-01',periods=6,freq='7D',tz='UTC')
    frame=pd.DataFrame({'decision_time':dates,'asset':['a']*6,'y':[.01,.02,.03,.02,.01,.03],
        'label_available_at':dates+pd.Timedelta(days=7),'sequence_index':range(6),
        'max_dependency_available_at':dates})
    raw=np.tile([2.750011574,4.,1.,1.],(6,2,1));raw[5,:,0]=3.750011574
    train,test=frame.iloc[:4],frame.iloc[4:];cutoff=dates[4]
    contract={'base_raw_columns':[1],'source_raw_columns':[[0,2,3]],'meta_raw_columns':[0,2,3],'source_value_columns':[[1]]}
    p=prepare_sia(train,test,raw,cutoff,SCHEMA,neural_contract=contract)
    assert p['normalizer_id']==asset_normalizer(train,cutoff)[1]
    assert p['test']['model_matrix'].shape==(2,2,12)
    assert p['test_source_valid'].dtype==bool and p['test_source_valid'].all()
    assert p['neural_args']['source_columns']==[[0,2,3,4,6,7,8,10,11]]
    assert np.array_equal(test.y.to_numpy(),np.array([.01,.03]))

def test_explicit_spec_schema_not_name_substring():
    from signalforge.v33_sia import schema_from_specs
    specs=[{'name':'numeric_release_age_trick','source':'macro'}]
    names=['numeric_release_age_trick','numeric_release_age_trick_release_age','numeric_release_age_trick_reference_age','numeric_release_age_trick_observed','numeric_release_age_trick_coverage','season_sin','season_cos','context_padding_indicator']
    schema=schema_from_specs(specs,names)
    assert schema[0].kind=='numeric' and schema[1].kind=='release_age'