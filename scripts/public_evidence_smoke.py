"""Verify only committed aggregate research evidence using the Python standard library.

This is deliberately NOT a refit, an independent replication of frozen runs, a
validation of point-in-time original publication clocks, or a financial backtest.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE = ROOT / "evidence" / "aggregates" / "overall_metrics.csv"
TRACKS = ("Main-A", "Nested-B")
MODELS = ("original_RGMF", "strong_I0", "SIA_v2", "BAR", "SIA_calibrated_v34")


def verify_public_metrics(csv_path: Path) -> dict[str, dict[str, float]]:
    """Validate the public five-model, two-track descriptive aggregate contract."""
    data: dict[tuple[str, str], dict[str, str]] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {
            "track", "model", "slice", "value", "rows", "distinct_dates",
            "assets", "seeds", "loss", "coverage90", "crossings",
        }
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError("Missing aggregate columns")
        for row in rows:
            if row["slice"] != "ALL" or row["value"] != "ALL":
                continue
            key = row["track"], row["model"]
            if key in data:
                raise ValueError(f"Duplicate overall row: {key}")
            if any(int(row[field]) != expected for field, expected in (
                ("rows", 1248), ("distinct_dates", 52),
                ("assets", 8), ("seeds", 3), ("crossings", 0)
            )):
                raise ValueError(f"Unexpected public evaluation grid: {key}")
            for field in ("loss", "coverage90"):
                value = float(row[field])
                if not math.isfinite(value) or not (0 < value < (1 if field == "coverage90" else math.inf)):
                    raise ValueError(f"Nonfinite or invalid {field}: {key}")
            data[key] = row

    expected_keys = {(track, model) for track in TRACKS for model in MODELS}
    if set(data) != expected_keys:
        raise ValueError(f"Unexpected model-track inventory: missing={sorted(expected_keys - set(data))}, extra={sorted(set(data) - expected_keys)}")

    results = {}
    for track in TRACKS:
        baseline = float(data[(track, "strong_I0")]["loss"])
        sia = float(data[(track, "SIA_v2")]["loss"])
        rgmf = float(data[(track, "original_RGMF")]["loss"])
        # Signed: negative means a lower (better) normalized pinball loss.
        signed_relative_pct = 100 * (sia / baseline - 1)
        against_rgmf_pct = 100 * (1 - sia / rgmf)
        coverage90_pct = 100 * float(data[(track, "SIA_v2")]["coverage90"])
        results[track] = {
            "strong_I0_NPL": baseline,
            "SIA_NPL": sia,
            "SIA_vs_strong_I0_pct": signed_relative_pct,
            "SIA_improvement_vs_RGMF_pct": against_rgmf_pct,
            "SIA_coverage90_pct": coverage90_pct,
        }

    if abs(results["Main-A"]["SIA_vs_strong_I0_pct"] - (-1.134)) > 0.01:
        raise ValueError("Main-A published comparator claim drifted")
    if abs(results["Nested-B"]["SIA_vs_strong_I0_pct"] - 0.493) > 0.01:
        raise ValueError("Nested-B published comparator claim drifted")
    if abs(results["Nested-B"]["SIA_improvement_vs_RGMF_pct"] - 15.13) > 0.02:
        raise ValueError("Nested-B RGMF comparator claim drifted")
    if abs(results["Main-A"]["SIA_coverage90_pct"] - 79.97) > 0.02:
        raise ValueError("Main-A descriptive coverage claim drifted")
    if abs(results["Nested-B"]["SIA_coverage90_pct"] - 78.77) > 0.02:
        raise ValueError("Nested-B descriptive coverage claim drifted")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    args = parser.parse_args()
    try:
        results = verify_public_metrics(args.aggregate)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Public evidence check FAILED: {exc}\n")
    print("Public aggregate check PASSED: 2 tracks x 5 model families")
    for track, metrics in results.items():
        print(
            f"{track}: I0 NPL={metrics['strong_I0_NPL']:.6f}, "
            f"SIA NPL={metrics['SIA_NPL']:.6f}, "
            f"SIA vs I0={metrics['SIA_vs_strong_I0_pct']:+.3f}%, "
            f"SIA 90% interval coverage={metrics['SIA_coverage90_pct']:.2f}%"
        )
    print("Scope: checks saved public summary data ONLY; no fit, new holdout, or certified PIT evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
