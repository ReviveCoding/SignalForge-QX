import numpy as np
import pytest
from signalforge.track_statistics import DateWeightedStatistical,calendar_recency,weighted_quantiles


def test_ewma_same_date_assets_have_identical_recency_and_ragged_date_mass():
    dates=['2022-01-07T23:00Z','2022-01-07T23:00Z','2022-01-14T23:00Z']
    decay=calendar_recency(dates);assert decay.tolist()==pytest.approx([.97,.97,1.])
    y=np.array([0.,10.,5.]);weights=np.array([.5,.5,1.])
    model=DateWeightedStatistical('ewma').fit(np.ones((3,1)),y,weights=weights,dates=dates)
    assert model.mean==pytest.approx(5.)
    reversed_within_date=DateWeightedStatistical('ewma').fit(np.ones((3,1)),y[[1,0,2]],weights=weights,dates=dates)
    assert model.mean==reversed_within_date.mean and np.array_equal(model.q,reversed_within_date.q)
    with pytest.raises(ValueError,match='calendar'):DateWeightedStatistical('ewma').fit(np.ones((3,1)),y)


def test_residual_quantiles_respect_date_weights_and_zero_mass_rows():
    values=np.array([-10.,0.,10.,1e6]);weights=np.array([1.,8.,1.,0.])
    assert weighted_quantiles(values,weights)[2]==pytest.approx(0.)
    first=DateWeightedStatistical('ridge',regularization=1.).fit(np.ones((3,1)),values[:3],weights=weights[:3])
    np.testing.assert_array_equal(first.residual_q,weighted_quantiles(values[:3]-first.mean_model.predict(np.ones((3,1))),weights[:3]))
