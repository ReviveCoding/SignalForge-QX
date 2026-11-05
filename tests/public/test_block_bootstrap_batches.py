import numpy as np
import pytest

from signalforge.metrics import paired_time_block_ci


@pytest.mark.parametrize("n,block,draws", [(1, 1, 101), (31, 8, 137), (10001, 8, 137)])
@pytest.mark.parametrize("seed", [0, 731, 902])
def test_block_bootstrap_keeps_every_seeded_draw(n, block, draws, seed, monkeypatch):
    values = np.random.default_rng(51).normal(size=n)
    rng = np.random.default_rng(seed)
    count = int(np.ceil(n / block))
    expected = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=count)
        positions = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        expected.append(values[positions].mean())
    quantile = np.quantile

    def verify_draws(actual, probabilities):
        np.testing.assert_array_equal(actual, expected)
        return quantile(actual, probabilities)

    monkeypatch.setattr(np, "quantile", verify_draws)
    result = paired_time_block_ci(values, block, draws, seed)
    assert result["n_dates"] == n and result["seed"] == seed
    assert result["mean_improvement"] == float(values.mean())
