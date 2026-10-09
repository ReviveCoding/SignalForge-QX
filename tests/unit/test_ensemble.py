import numpy as np
import pytest
from signalforge.ensemble import FrozenEnsemble,TargetUnits


def test_frozen_members_weights_and_separate_mean():
    ensemble=FrozenEnsemble(['s11','s37','s71'])
    pred={s:(np.array([10+i]),np.array([[0,1,2,3,4]])+i) for i,s in enumerate(ensemble.member_ids)}
    mean,q=ensemble.predict(pred)
    assert mean[0]==11 and q[0,2]==3
    with pytest.raises(ValueError):ensemble.predict({k:v for k,v in pred.items() if k!='s71'})
    with pytest.raises(ValueError):FrozenEnsemble(['s11','s11'])
    with pytest.raises(ValueError):FrozenEnsemble(['s11','s37'],[-1,2])


def test_inverse_units_before_ensemble():
    units=TargetUnits(2,10,'million_barrels')
    mean,q=units.inverse([.2],[[0,1,2,3,4]])
    assert mean[0]==4 and q[0,2]==22
