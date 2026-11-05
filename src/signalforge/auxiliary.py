"""Registered reconstructed EIA development comparison, never a full study/final substitute."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import auxiliary_rows
from .experiments import TrialLedger
from .features import TrainTransform, purged_training, shared_target_scale
from .metrics import quantile_loss
from .models import QUANTILES, CudaTree, NeuralCUDA, Statistical
from .runtime import atomic_json, code_hash, commit_bundle, digest, gpu_lease, now, validate_bundle

FAMILIES = [
    "historical",
    "ewma",
    "ridge",
    "linear_quantile",
    "lightgbm",
    "xgboost",
    "mlp",
    "gru",
    "gru_d",
    "tft",
    "mixed_frequency_shrinkage",
    "rgmf_linear",
    "rgmf_gru",
    "rgmf_transformer",
    "lgb_no_age",
    "lgb_no_coverage",
    "rgmf_residual_gru",
    "ssl_gru",
]


def sequences(frame, lookback=26):
    frame = frame.sort_values("decision_time").reset_index(drop=True)
    rows = []
    x = []
    for i in range(lookback - 1, len(frame)):
        if not np.isfinite(frame.iloc[i].y):
            continue
        context = frame.iloc[i - lookback + 1 : i + 1]
        decisions = pd.to_datetime(context.decision_time, utc=True)
        if (decisions.diff().dropna() > pd.Timedelta(days=10)).any():
            continue
        if (
            pd.to_datetime(context.max_dependency_available_at, utc=True)
            > pd.Timestamp(frame.iloc[i].decision_time)
        ).any():
            raise ValueError("Future feature lineage")
        rows.append(frame.iloc[i].to_dict())
        x.append(np.array(context.x.tolist()))
    result = pd.DataFrame(rows)
    result["sequence_index"] = np.arange(len(result))
    return result, np.array(x)


def fit_predict(
    family, x, y, scale, test_x, seed=11, regularization=1, neural_trial=None, checkpoint_path=None
):
    """All architectures receive the same context/metadata; CPU baselines remain explicit."""
    before = time.monotonic()
    if family in {"historical", "ewma", "ridge", "linear_quantile", "mixed_frequency_shrinkage"}:
        m = Statistical(family, regularization).fit(x.reshape(len(x), -1), y)
        mean, q = m.predict(test_x.reshape(len(test_x), -1))
    elif family in {"lightgbm", "xgboost", "lgb_no_age", "lgb_no_coverage"}:
        backend = "lightgbm" if family.startswith("lgb_") else family
        m = CudaTree(backend, rounds=100, seed=seed, regularization=regularization).fit(
            x.reshape(len(x), -1), y
        )
        mean, q = m.predict(test_x.reshape(len(test_x), -1))
    else:
        args = {}
        if family.startswith("rgmf"):
            f = x.shape[-1] // 2
            # Base uses autoregressive commercial stocks/change and deterministic seasonality.
            base = [0, 1, 12, 13, f, f + 1, f + 12, f + 13]
            source = [i for i in range(x.shape[-1]) if i not in base]
            args = {
                "base_columns": base,
                "source_columns": [source],
                "meta_columns": [10, 11, f + 10, f + 11],
            }
        neural_grid = {
            0.01: (16, 0.001),
            0.1: (16, 0.003),
            1: (32, 0.001),
            10: (32, 0.003),
            100: (64, 0.001),
            1000: (64, 0.003),
        }
        width, lr = neural_trial or neural_grid[regularization]
        m = NeuralCUDA(family, width=width, epochs=100, seed=seed, lr=lr, **args)
        mask = np.ones_like(x, dtype=bool)
        ages = np.zeros_like(x)
        m.fit(x, y, scale, observed=mask, elapsed=ages, checkpoint_path=checkpoint_path)
        mean, q = m.predict(test_x, observed=np.ones_like(test_x, dtype=bool), elapsed=np.zeros_like(test_x))
    return m, mean, q, time.monotonic() - before


def run(repo, runtime):
    import subprocess
    import sys

    # Independent positioning acquisition remains useful even with blocked prices.
    # EIA acquisition has finished before this entry point; no extra source workers.
    if (repo / "reports/source_document_verification.json").exists():
        subprocess.run([sys.executable, "scripts/acquire_cftc_development.py"], check=False, timeout=1320)
    from .development import run_auxiliary

    return run_auxiliary(repo, runtime)
