import json,numpy as np,pytest
from calibration_v34.serialization_r1.adapter import native,corrected_fit
from calibration_v34.test_contract import data
from calibration_v34.api import fit
@pytest.mark.parametrize('kind',['identity','intercept','cqr'])
def test_numpy_support_regression(kind):
    raw=fit(data(),kind,'2020-01-01T00:00Z');fixed=corrected_fit(data(),kind,'2020-01-01T00:00Z');assert fixed['supported']==raw['supported'];assert fixed['offset_normalized']==raw['offset_normalized'];assert all(type(v)==bool for v in fixed['supported']);json.dumps(fixed,allow_nan=False)
def test_no_numeric_change():assert native({'x':np.float64(1.25),'b':np.bool_(True),'n':np.int64(52)})=={'x':1.25,'b':True,'n':52}