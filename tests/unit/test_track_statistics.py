import numpy as np
import pytest

from signalforge.track_statistics import DateWeightedStatistical, calendar_recency, weighted_quantiles


def test_ewma_same_date_assets_have_identical_recency_and_ragged_date_mass():
    dates = ["2022-01-07T23:00Z", "2022-01-07T23:00Z", "2022-01-14T23:00Z"]
    decay = calendar_recency(dates)
    assert decay.tolist() == pytest.approx([0.97, 0.97, 1.0])
    y = np.array([0.0, 10.0, 5.0])
    weights = np.array([0.5, 0.5, 1.0])
    model = DateWeightedStatistical("ewma").fit(np.ones((3, 1)), y, weights=weights, dates=dates)
    assert model.mean == pytest.approx(5.0)
    reversed_within_date = DateWeightedStatistical("ewma").fit(
        np.ones((3, 1)), y[[1, 0, 2]], weights=weights, dates=dates
    )
    assert model.mean == reversed_within_date.mean and np.array_equal(model.q, reversed_within_date.q)
    with pytest.raises(ValueError, match="calendar"):
        DateWeightedStatistical("ewma").fit(np.ones((3, 1)), y)


def test_residual_quantiles_respect_date_weights_and_zero_mass_rows():
    values = np.array([-10.0, 0.0, 10.0, 1e6])
    weights = np.array([1.0, 8.0, 1.0, 0.0])
    assert weighted_quantiles(values, weights)[2] == pytest.approx(0.0)
    first = DateWeightedStatistical("ridge", regularization=1.0).fit(
        np.ones((3, 1)), values[:3], weights=weights[:3]
    )
    np.testing.assert_array_equal(
        first.residual_q,
        weighted_quantiles(values[:3] - first.mean_model.predict(np.ones((3, 1))), weights[:3]),
    )
