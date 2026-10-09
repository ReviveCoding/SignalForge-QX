# SignalForge-QX v3.8.1-research

**Public research pre-release | Documentation completeness patch**

This release adds **seven missing aggregate-only technical reports** that were excluded from the first v3.8.0 publication by an overbroad reports/ Git ignore pattern. Those reports already existed in the privacy-reviewed publication staging directory; this patch commits them and anchors the ignore pattern to the private top-level reports directory. The original v3.8.0 annotated tag is preserved, not rewritten.

## Included in the public snapshot

- Python source for source-aware quantitative forecasting, SIA, BAR, point-in-time evidence contracts, mature OOF calibration, official-source provenance qualification, and post-hoc model-risk evaluation.
- **Seven public technical reports**, five aggregate numerical tables, and three graphical diagnostics.
- Offline GitHub Actions Python syntax and JSON structural validation.

## Retrospective development measurements

| Normalized pinball loss (smaller is better) | Main-A 2020 | Nested-B 2022 |
|---|---:|---:|
| Strong I0 comparator | 0.498910 | **0.272733** |
| Source-age SIA | **0.493252** | 0.274078 |
| BAR | 0.500807 | 0.274928 |
| Calibrated SIA | 0.494581 | 0.275097 |

SIA reduced NPL by **1.134%** versus the strongest baseline on Main-A; it was **0.493% worse** than the strongest baseline on Nested-B, despite a **15.13%** relative improvement against the older RGMF there. BAR and added quantile calibration were not promoted.

Research reproducibility evidence: 120 genuine CUDA OOF SIA fits and saved checkpoint replays, 12,168 OOF forecast rows, 12,480 aligned development forecasts, 4,737 historical dry-run maturity-aware monitoring replay rows, and 98 passing scoped v3.8 diagnostics. These measurements **were not re-executed from the public snapshot** and v3.8 added no new model training.

## Nonqualification and data boundaries

This is a pre-release of **retrospective Tier-B reconstructed-clock research**, not a production release or independently qualified financial model. Certified original first-public Tier-A events: **0**. Fully qualified P1 economic sessions: **0/27,088**. Actual independently frozen future market forecasts and mature forward observations: **0**. No verified real trading P&L, Sharpe, alpha, operational calibration FPR, independent reviewer signoff, or final unsealing.

Vendor datasets, private model/checkpoint bundles, internal compute ledgers, API tokens, unpublished event-version histories, private prompts and raw price/label panels are excluded. Local machine username occurrences in public code were redacted, so public code is **not byte-identical to internal frozen research source**.

See [README](https://github.com/ReviveCoding/SignalForge-QX/blob/v3.8.1-research/README.md), [reproducibility](https://github.com/ReviveCoding/SignalForge-QX/blob/v3.8.1-research/docs/REPRODUCIBILITY.md), [publication boundaries](https://github.com/ReviveCoding/SignalForge-QX/blob/v3.8.1-research/docs/PUBLICATION_BOUNDARY.md), and [v3.8 model-risk report](https://github.com/ReviveCoding/SignalForge-QX/blob/v3.8.1-research/evidence/reports/V38_INCREMENTAL_MODEL_RISK_AND_CALIBRATION_REPORT.md).
