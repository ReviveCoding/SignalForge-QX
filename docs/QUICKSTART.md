# Public quickstart

This is a **research publication snapshot**. The commands below do not access
private datasets, trained weights, reserved labels, live market feeds or paid APIs.

## Prerequisites

- Git and Python 3.11 or newer.
- For optional unit tests: internet access for the first installation of Python
  dependencies (NumPy, pandas and pytest). Runtime checks use synthetic fixtures.

## 1. Clone and verify the published aggregate results

```sh
git clone https://github.com/ReviveCoding/SignalForge-QX.git
cd SignalForge-QX
python scripts/public_evidence_smoke.py
```

On Windows PowerShell, `py -3.11 scripts/public_evidence_smoke.py` works
if the Python launcher is installed.

The script uses **only the Python standard library** and reads
`evidence/aggregates/overall_metrics.csv`. It checks two tracks, five model
families, evaluation-grid counts, directional comparator results and descriptive
interval coverage. It should print **Public aggregate check PASSED**.

**It does not train models, independently reproduce the original CUDA runs,
certify Tier-A source clocks, or validate future performance.** Results are
checked against published aggregate rows, not against raw predictions.

## 2. Run real model-primitive unit tests (optional)

```sh
python -m pip install -e ".[public-test]"
python -m pytest -q tests/unit/test_starter_primitives.py tests/public
```

On PowerShell, replace `python` with `py -3.11` if needed. You may also
use a local Python virtual environment. No API keys or NVIDIA GPU are needed.
These tests execute actual source functions for point-in-time release selection,
observation maturity, quantile loss, block bootstrap, and public-evidence checks
on synthetic fixtures and committed summaries.

This is **not** the original full training or full-repository verification suite.
Other training/integration tests require original dependencies, artifacts,
registries, authorization and sometimes CUDA.

## 3. Optional additional publication checks

```sh
python scripts/check_publication_boundary.py
python -m compileall -q src scripts tests policy_v3 studies_v3 diagnostics_v3 p1_pit_readiness_v34 calibration_v34 qualification_v35 data_recovery_v36 evidence_procurement_v37 model_risk_v38
```

The publication boundary checker scans Git-tracked filenames and a small set
of credential signatures. It cannot prove the absence of all secrets, private
information or restricted material. GitHub's independent security controls and
a human review remain important.

## What's next

- [Architecture overview](ARCHITECTURE.md)
- [Version convention](VERSIONING.md)
- [Reproducibility boundaries](REPRODUCIBILITY.md)
- [Publication limitations](PUBLICATION_BOUNDARY.md)
- [Public technical evidence](../evidence/)

**No open-source reuse license has been granted.** Public visibility is not
the same as permission to redistribute or use this code in another product.
