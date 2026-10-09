# SignalForge-QX continuation report — 2026-10-07

State: COMPLETED_ADMITTED_DEVELOPMENT_SCIENTIFIC_GATES_BLOCKED. Reconstructed Tier-B prediction development only. Strict final and P1 claims are not permitted.

## Executed and integrity-verified

| Branch | Actual result | Evidence |
|---|---|---|
| Source extension | Main-A I0–I3; Nested-B I0–I4 ready; zero blocked sets; base I0 parity retained | track_source_input_extension_v311.json; track_input_qualification_v311.json |
| CPU | Main-A 1,925; Nested-B 924; total 2,849 immutable metric/model pairs valid; no failures | track_input_qualification_v311.json; successor_continuation_terminal.json |
| Successor pilot | Six successful fits per track; 12 total | successor_gpu_pilot_Main_A.json; successor_gpu_pilot_Nested_B.json |
| Measured-bill admission | Outcome blind; fixed 540-result plan admitted; 2× safety projection 13491.79975985855 seconds | successor_gpu_bill_admission.json; successor_execution_plan.json |
| Full successor study | Main-A 360; Nested-B 180; all 540 pairs valid, zero blocked folds; ten exact pilot reuses | successor_gpu_full.json; successor_gpu_integrity.json; successor_track_checkpoint_Main_A.json; successor_track_checkpoint_Nested_B.json |
| Calibration | Main-A 44 candidates; Nested-B 55; 25 distinct mature dates; median correction, identity unsupported tails | calibration_Main_A.json; calibration_Nested_B.json |
| Development analysis | 99 tables; 97 paired contrasts; seeds averaged within dates; 4/8/13-week blocks and full multiplicity family retained | track_development_analysis.json; successor_development_interpretation.json |

Six GPU families: lightgbm, xgboost, mlp, gru, rgmf_linear, rgmf_gru. Seeds 11, 37, 71. Main-A 2018–2022 and I0–I3; Nested-B 2021–2022 and I0–I4. Settings and execution plan remained frozen. The registered 4,140-fit HPO bill was not admitted (projected cost exceeds the immutable ceiling); these fixed-setting results are not a tuned-study substitution. Transformers were not admitted: Main-A training dates 363–571; Nested-B 39–92; registered minimum 600.

## Cost and immutable budgets

| Ledger | Charged seconds | Ceiling | Entries |
|---|---:|---:|---:|
| parent_ledger | 42907.269684 | 43200.0 | 3610 |
| pilot_ledger | 490.303528 | 3600.0 | 18 |
| full_ledger | 5234.511645 | 21600.0 | 597 |

Parent digest: `2705b3d5e54f99b4d5813cdab40a3de9ab9dfe849315f46f95075c47953748cb`. Unchanged. No refund, reset, reclassification or ceiling increase.
Full accounting: 530 canonical charged fits (4646.809818s), 54 calibration inference charges (10.352082s), 12 retained historical successful charges (277.349745s), and retained 300s crash reservation. See successor_compute_accounting.json for track/family distribution.

## Findings and boundaries

Information additions are not uniformly beneficial. LightGBM macro increments worsened loss on both tracks; subsequent source increments had mixed directions. Historical/recurrent methods remain competitive in the fixed-setting development table. Nested-B I4 MLP produced very large finite losses; these results were retained. No outcome-driven retuning, seed selection or architecture-plan change.
Calibration changes are in-sample diagnostics, not out-of-sample improvement claims. Tails lack registered support and stay identity. Exact candidate/ensemble IDs, fit cutoffs, immutable prediction hashes and per-quantile support are recorded.
H2/H3 and component retraining/outage experiments remain unexecuted: selection-frozen candidate identity, track-specific frozen corruption schedule and admitted execution bill are absent. Their contrasts remain in the multiple-comparison family.

## Genuine remaining gates

| Branch | State / exact missing evidence |
|---|---|
| Strict PIT / Tier A | Zero authenticated Tier-A events. Per-version original release payload and first-public clocks missing for market, macro, CFTC, EIA and N-PORT; exact requests in strict_pit_final_gate_audit.json |
| P1 | Raw opens present; complete corporate actions, ex/payment clocks, splits and qualified cash benchmark absent; Tiingo actions 403 entitlement retained in keyed_source_access.json |
| Auxiliary-C | Independent 2023 EIA source arithmetic discrepancy remains BLOCKED_SOURCE_CONSISTENCY; eia_2023_calibration_source_audit.json |
| Scientific freeze | Genuine strict-source/candidate/postprocess gates incomplete; freeze receipt absent |
| Reserved final | Sealed. Existing scientific authorization does not replace freeze; zero reserved batches executed |
| Prospective | Operational freeze absent; no forecast issued or backdated; not yet a future-outcome waiting forecast |

## Repairs and validation

After the original worker naturally exited on an NTFS reader-lock EACCES, reproduced the WSL/Windows atomic replacement failure. Minimal bounded retry publishes the same flushed temporary bytes and still fails closed on persistent/structural errors. Corrected supervisor full-ledger display. No backend/math/grid/seed changes. Existing watcher resumed from durable cache.
After full-study exit, fixed stale phase presentation that let an old blocked receipt override completed GPU/calibration evidence. Regression reproduced 2 failures, focused continuation/reporting suite 28 passed. Atomic/recovery focused suite 41 passed.
Final full validation: 501 passed (51 handoff, 392 unit, 40 CUDA, 18 real-source), zero failures/errors/skips, inputs_unchanged=true; reports/test_execution.json. Source `01a398d12694215bfee63bea6f7cb8acbad03d090a0d64f9a449bae7293c3bb5`; qualification `6b38cefa302e59eb1d186e1ecb7a6ed531d2eada86356883bbf19bae148b7ebd`.
GPU execution source `52b00ca61b21f34aff7dee443f4f1093593555a0d563d32cd440af8df754a6ca`. Later source change was receipt presentation only. Original admission hash `4affde9808b6f7edab58837a31c4ec33c8b639170920a9f1aa9402d80521b2f6` survives in immutable report snapshots listed in successor_continuation_terminal.json. All failure receipts, model bundles and ledger entries remain preserved.

Changed source files: src/signalforge/runtime.py; src/signalforge/successor_runtime.py; src/signalforge/track_status.py; tests/unit/test_atomic_receipt_retry.py; tests/unit/test_continuation_phase_status.py. Updated EXECUTION_PLAN.md, IMPLEMENTATION_STATUS.json and current reports/checkpoints/cards/evidence. No source acquisition, CPU refitting or duplicate full GPU run.

## Final state

Implementation complete: **false**. Reserved evaluation complete: **false**. Strict final scientific claims permitted: **false**. Development branch complete under the admitted fixed-setting protocol; scientific and external-evidence blockers remain explicit.
GPU watcher idle; live progress/recovery report RESERVED FINAL SEALED. No active training/calibration worker. Native Codex tool output observed receipts and ran authorized postprocessing; it did not attach old buffered worker stdout.

## Measured successful pilot fit charges

| Family | Main-A seconds | Nested-B seconds |
|---|---:|---:|
| lightgbm | 16.154425 | 12.410022 |
| xgboost | 5.019375 | 5.016971 |
| mlp | 20.768859 | 7.427791 |
| gru | 16.632663 | 6.198099 |
| rgmf_linear | 17.942775 | 4.881102 |
| rgmf_gru | 14.830680 | 6.231787 |
