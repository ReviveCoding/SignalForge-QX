import numpy as np
import pytest
from signalforge.frozen_processing import candidate_ensemble_id,validate_frozen_postprocessing,apply_frozen_postprocessing


def fixture():
    candidate={'candidate_id':'ridge','track':'Main-A','members':[{'member_id':'11','relative_bundle':'models/one','receipt_id':'a'*64}],'weights':[1.]}
    identity=candidate_ensemble_id(candidate)
    calibration={'candidates':{'ridge':{'ensemble_id':identity,'corrections':[0.,0.,-10.,0.,0.],'quantiles':[.05,.1,.5,.9,.95],
        'fit_window':{'start':'2023-07-01T00:00Z','end':'2023-12-31T23:59Z','max_label_available_at':'2023-12-30T00:00Z'},'fit_data_id':'b'*64,
        'support':{str(q):{'n_dates':26,'status':'CORRECTION' if q==.5 else 'IDENTITY_INSUFFICIENT_SUPPORT'} for q in [.05,.1,.5,.9,.95]}}}}
    return candidate,{'candidate_ensemble_ids':{'ridge':identity}},calibration


def test_frozen_calibration_bound_to_receipts_and_preserves_raw():
    candidate,ensemble,calibration=fixture();assert validate_frozen_postprocessing([candidate],ensemble,calibration)
    raw=np.array([[-2.,-1.,0.,1.,2.]])
    processed=apply_frozen_postprocessing(candidate,[.3],raw,calibration)
    assert np.array_equal(processed['raw_quantiles'],raw)
    assert processed['calibration_crossing_rows']==1 and np.all(np.diff(processed['quantiles'])>=0)
    assert processed['mean'][0]==.3
    candidate['members'][0]['receipt_id']='c'*64
    with pytest.raises(PermissionError,match='model receipts'):validate_frozen_postprocessing([candidate],ensemble,calibration)


def test_frozen_calibration_rejects_future_maturity_and_insufficient_tail_correction():
    candidate,ensemble,calibration=fixture();record=calibration['candidates']['ridge']
    record['corrections'][0]=1.
    with pytest.raises(PermissionError,match='identity'):validate_frozen_postprocessing([candidate],ensemble,calibration)
    record['corrections'][0]=0.;record['fit_window']['max_label_available_at']='2024-01-02T00:00Z'
    with pytest.raises(PermissionError,match='maturity'):validate_frozen_postprocessing([candidate],ensemble,calibration)


def test_calibration_cannot_claim_tail_support_from_26_dates():
    candidate,ensemble,calibration=fixture();record=calibration['candidates']['ridge']
    record['support']['0.05']['status']='CORRECTION'
    with pytest.raises(PermissionError,match='distinct-date support'):validate_frozen_postprocessing([candidate],ensemble,calibration)


def test_frozen_prediction_resume_rejects_grid_and_processed_forecast_change():
    from signalforge.final import validate_prediction_resume
    candidate,ensemble,calibration=fixture();grid=[{'decision_time':'2024-01-05T23:00Z','asset':'SPY'}]
    raw=np.array([[-2.,-1.,0.,1.,2.]]);processed=apply_frozen_postprocessing(candidate,[.3],raw,calibration)
    record={'candidate_id':'ridge','track':'Main-A','grid':grid,'mean':[.3],'raw_quantiles':raw.tolist(),
        'quantiles':processed['quantiles'].tolist(),'before_rearrangement':processed['before_rearrangement'].tolist(),
        'ensemble_id':processed['ensemble_id'],'calibration_crossing_rows':1,'native_prediction_coverage':1.,'failure_type':None,'fallback_candidate_id':'ridge'}
    assert validate_prediction_resume(record,candidate,candidate,grid,calibration)
    with pytest.raises(PermissionError,match='grid'):validate_prediction_resume(record,candidate,candidate,[{**grid[0],'asset':'QQQ'}],calibration)
    record['quantiles'][0][0]=100.
    with pytest.raises(PermissionError,match='processing'):validate_prediction_resume(record,candidate,candidate,grid,calibration)
