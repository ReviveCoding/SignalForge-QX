"""Data-free checks for the public snapshot's numerical and metadata contracts."""
import csv
import tomllib
from pathlib import Path

import pytest
import signalforge
from public_evidence_smoke import DEFAULT_AGGREGATE, verify_public_metrics

ROOT = Path(__file__).resolve().parents[2]


def _mutated_copy(tmp_path, mutate):
    with DEFAULT_AGGREGATE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    mutate(rows)
    destination = tmp_path / "overall_metrics.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def test_public_aggregate_claims_are_recomputable_from_committed_summaries():
    result = verify_public_metrics(DEFAULT_AGGREGATE)
    assert result["Main-A"]["SIA_vs_strong_I0_pct"] < 0
    assert result["Nested-B"]["SIA_vs_strong_I0_pct"] > 0
    assert all(result[track]["SIA_coverage90_pct"] < 90 for track in result)


def test_duplicate_summary_is_rejected(tmp_path):
    def duplicate(rows):
        rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="Duplicate overall"):
        verify_public_metrics(_mutated_copy(tmp_path, duplicate))


def test_missing_model_is_rejected(tmp_path):
    def drop_one(rows):
        rows.pop()
    with pytest.raises(ValueError, match="Unexpected model-track inventory"):
        verify_public_metrics(_mutated_copy(tmp_path, drop_one))


def test_quantile_crossing_or_grid_drift_is_rejected(tmp_path):
    def corrupt(rows):
        rows[0]["crossings"] = "1"
    with pytest.raises(ValueError, match="Unexpected public evaluation grid"):
        verify_public_metrics(_mutated_copy(tmp_path, corrupt))


def test_python_package_versions_agree_without_retagging_research_history():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["version"] == signalforge.__version__
