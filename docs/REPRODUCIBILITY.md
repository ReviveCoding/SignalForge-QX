# Reproducibility and local execution

SignalForge-QX was developed with Python, PowerShell 7, WSL and a local NVIDIA RTX 4090 Laptop GPU. The public repository is a **privacy-redacted code snapshot**, not the original immutable artifact directory.

Original execution data, model/checkpoint binaries, qualified source event versions and internal ledgers are deliberately excluded. Consequently a new public checkout **cannot reproduce all reported numerical scores** without those separately authorized files, exact dependencies and registration metadata. Paths containing **USERNAME** need to be adapted to the reader's machine.

## Offline Python syntax smoke test

With Python 3.11+ in the repository root, run:

    python -m compileall -q src scripts tests policy_v3 studies_v3 diagnostics_v3 p1_pit_readiness_v34 calibration_v34 qualification_v35 data_recovery_v36 evidence_procurement_v37 model_risk_v38

GitHub Actions verifies only Python syntax and JSON file parsing. This check does **not** validate CUDA fitting, scientific point-in-time information quality, model accuracy, or historic experimental parity.

## Public aggregate research evidence

- [SIA development execution](../evidence/reports/V33_MINIMAL_EXECUTION_RESULTS_V2.md)
- [Genuine SIA OOF and negative calibration](../evidence/reports/V34_CALIBRATION_INTERPRETATION.md)
- [v3.8 incremental model-risk findings](../evidence/reports/V38_INCREMENTAL_MODEL_RISK_AND_CALIBRATION_REPORT.md)
- [Five-family aggregate NPL](../evidence/aggregates/overall_metrics.csv)
- [Date-block sensitivity](../evidence/aggregates/paired_date_block_bounds.csv)
- [Maturity-aware monitoring replay](../evidence/aggregates/monitor_summary.csv)

These retrospective evaluation cohorts use reconstructed Tier-B source timing; they are not a new or unseen independent holdout. Vendor permissions, authenticated original release clocks, and future verified time-series data remain missing. Results cannot be promoted to a live financial product based on this snapshot.
