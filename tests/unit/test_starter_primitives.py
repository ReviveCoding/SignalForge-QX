import numpy as np
import pandas as pd
import pytest
from signalforge.pit import asof_snapshot, nport_external_flow, mature_training_rows
from signalforge.metrics import quantile_loss, paired_time_block_ci


def events():
    return pd.DataFrame([
        dict(entity="A", source="s", field="x", reference_time="2024-01-02T00:00:00Z", available_at="2024-01-05T20:30:00Z", value=1, pit_tier="A"),
        dict(entity="A", source="s", field="x", reference_time="2024-01-09T00:00:00Z", available_at="2024-01-12T20:30:00Z", value=2, pit_tier="A"),
        dict(entity="A", source="s", field="x", reference_time="2024-01-02T00:00:00Z", available_at="2024-01-15T20:30:00Z", value=9, pit_tier="B"),
    ])


def test_future_release_excluded():
    out = asof_snapshot(events(), "2024-01-10T23:00:00Z")
    assert out.value.tolist() == [1]


def test_old_revision_does_not_replace_new_reference():
    assert asof_snapshot(events(), "2024-01-16T23:00:00Z").value.tolist() == [2]


def test_latest_known_revision_of_same_reference():
    e = events().iloc[[0,2]]
    assert asof_snapshot(e, "2024-01-16T23:00:00Z").value.tolist() == [9]


def test_strict_filters_uncertified_versions():
    e = events().iloc[[0,2]]
    assert asof_snapshot(e, "2024-01-16T23:00:00Z", strict=True).value.tolist() == [1]


def test_naive_decision_rejected():
    with pytest.raises(ValueError):
        asof_snapshot(events(), "2024-01-16")


def test_naive_available_rejected():
    e = events(); e.loc[0, "available_at"] = "2024-01-05"
    with pytest.raises(ValueError):
        asof_snapshot(e, "2024-01-16T00:00:00Z")


def test_duplicate_version_rejected():
    e = pd.concat([events(), events().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        asof_snapshot(e, "2024-01-16T00:00:00Z")


def test_before_all_releases_empty():
    assert asof_snapshot(events(), "2024-01-01T00:00:00Z").empty


def test_source_ages_differ():
    out = asof_snapshot(events(), "2024-01-10T23:00:00Z")
    assert out.reference_age_days.iloc[0] > out.release_age_days.iloc[0] >= 0


def test_observed_future_reference_rejected():
    e = events(); e.loc[0,"reference_time"] = "2024-02-01T00:00:00Z"
    with pytest.raises(ValueError):
        asof_snapshot(e, "2024-02-02T00:00:00Z")


def test_external_flows_and_missingness():
    out = nport_external_flow([10, 5, np.nan], [4, 8, 2])
    np.testing.assert_allclose(out[:2], [6, -3]); assert np.isnan(out[2])


def test_negative_gross_flow_rejected():
    with pytest.raises(ValueError): nport_external_flow([-1], [2])


def test_flow_shape_rejected():
    with pytest.raises(ValueError): nport_external_flow([1,2], [2])


def test_label_publication_maturity():
    f = pd.DataFrame(dict(decision_time=["2024-01-01T00:00:00Z"]*2,
                         label_end=["2024-01-08T00:00:00Z"]*2,
                         label_available_at=["2024-01-09T00:00:00Z", "2024-01-20T00:00:00Z"]))
    assert len(mature_training_rows(f, "2024-01-10T00:00:00Z")) == 1


def test_invalid_label_publication_rejected():
    f = pd.DataFrame(dict(decision_time=["2024-01-01T00:00:00Z"], label_end=["2024-01-08T00:00:00Z"],
                         label_available_at=["2024-01-07T00:00:00Z"]))
    with pytest.raises(ValueError): mature_training_rows(f, "2024-01-10T00:00:00Z")


def test_pinball_perfect():
    np.testing.assert_allclose(quantile_loss([2], [[2,2,2]], [.1,.5,.9]), [0])


def test_pinball_median_absolute_half():
    np.testing.assert_allclose(quantile_loss([2,0], [[0],[2]], [.5]), [1,1])


def test_pinball_train_scale():
    np.testing.assert_allclose(quantile_loss([2], [[0]], [.5], scale=2), [.5])


def test_pinball_invalid_q():
    with pytest.raises(ValueError): quantile_loss([2], [[0,1]], [.9,.1])


def test_pinball_bad_scale():
    with pytest.raises(ValueError): quantile_loss([2], [[0]], [.5], scale=0)


def test_block_bootstrap_constant():
    out = paired_time_block_ci(np.ones(40), block_length=8, n_boot=100)
    assert out["lower_95"] == out["upper_95"] == out["mean_improvement"] == 1


def test_bootstrap_reproducible():
    x = np.arange(40)/100
    assert paired_time_block_ci(x, n_boot=100) == paired_time_block_ci(x, n_boot=100)


def test_bootstrap_insufficient_dates():
    with pytest.raises(ValueError): paired_time_block_ci([1,2], block_length=8)


def test_asof_mixed_iso_precision_remains_aware_and_naive_rejected():
    e=events().iloc[:2].copy()
    e.loc[e.index[0],'reference_time']='2024-01-01T00:00:00+00:00'
    e.loc[e.index[1],'reference_time']='2024-01-02T23:59:59.999999999+00:00'
    e.loc[e.index[0],'available_at']='2024-01-03T00:00:00+00:00'
    e.loc[e.index[1],'available_at']='2024-01-04T12:00:00.123456+00:00'
    out=asof_snapshot(e,'2024-01-05T00:00:00Z')
    assert not out.empty
    bad=e.copy();bad.loc[bad.index[0],'reference_time']='2024-01-01T00:00:00'
    with pytest.raises(ValueError,match='timezone-aware'):
        asof_snapshot(bad,'2024-01-05T00:00:00Z')
