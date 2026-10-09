# Minimal v3.3 SIA execution v2 — actual results

Six registered full CUDA fits completed. Pilot and full study have separate ledgers. No final qualification.

| Track/year | New SIA normalized pinball | Old RGMF | Strong baseline | Raw pinball | p99 / max absolute forecast |
|---|---:|---:|---:|---:|---|
| Main-A:2020 | 0.49325164 | 0.50061395 | 0.49891034671070106 | 0.01759421 | 0.342962 / 0.368463 |
| Nested-B:2022 | 0.27407759 | 0.32294800 | 0.27273327972034916 | 0.00958960 | 0.333174 / 0.457787 |

Pilot charge: 42.345139/300 seconds. Full six-fit charge: 81.602207/1800 seconds.
BAR: DEFERRED_PREDECLARED_SIGNAL_RULE_NOT_MET. Rule requires >=5% gain versus both matched controls on both folds and every seed better, with zero catastrophes/crossings; frozen before new outcomes.
All original source/357 baseline file hashes and three GPU ledgers remain unchanged. All rows/seeds retained. No reserved access. Per-asset/per-seed/calibration/catastrophe diagnostics and control receipt lineage are in v33_minimal_results_v2.json and immutable data bundles.
Historical overall-development references are not paired-fold results. Calibration here is empirical quantile coverage, not authenticated calibrated final forecasts.
ABC/full HPO/36-fit/v3.2/source-certification remain deferred. Strict PIT/P1/freeze/final gates unchanged.

Execution/validation notes:
- 70 current scoped CPU/admission/prequential tests passed. Seven historical minimal-v1 snapshot tests remain preserved with their receipt; they bind the old proposal source hash and are not execution-v2 qualification. Original full520 receipt remains unchanged and is not reused for v3.3.
- CUDA integration used actual development inputs and explicit SIA observed/missing/OOD channels and raw-value source-validity masks; fit/predict and checkpoint reload were checked. Each of six full fits also reloaded identically.
- Two 2-epoch CUDA integration invocations (one saved successfully before an event-output error), two 100-epoch cost pilots, and six 100-epoch full fits: ten training invocations total. The original completion-event TypeError and its successful model/charge were preserved. Repair1 changed telemetry only, froze source-bound revisionv2r1 and regression-tested; no charge reset/refund.
- Resource admission rejected transient ownership/utilization twice before any full fit; bounded observations then met the unchanged resource gate. No competing process killed, no global environment changed. The independent integrity command initially omitted the existing library path and hit a SQLite import error; rerunning with the configured process-local path passed, without code/model changes.
- Independent NumPy recomputation matches stored raw/normalized scores; all2496 forecast rows retained (1248 per track =52 dates x8 assets x3 seeds). Seeds are repeated fits, not independent dates. Exact old evaluation IDs were reconstructed; both frozen strong controls are I0 baselines, old RGMF uses I3/I4.
- Coverage diagnostics are not calibrated qualification: Main q0.95 coverage0.880609; Nested q0.10 coverage0.264423. Ordered finite forecasts do not establish tail calibration. No calibrated final claim.
- Original357 baseline files and all384 pre-existing isolated files preserved; original three ledgers identical. All new accounting is ext4 isolated runtime. v1/36-fit/v3.2 frozen records unchanged.
- BAR criterion was frozen before pilot outcomes: >=5% improvement against both controls on both folds and each seed better, zero catastrophes/crossings. Main improvements1.47%/1.13% fall below threshold, and Nested baseline loss worsens0.49%; therefore BAR not run. No ABC/HPO/other deferred study launched.
