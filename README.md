# SignalForge-QX

**Evidence-gated quantitative forecasting research with source-age-aware models, reproducible GPU experiments, probabilistic calibration, and model-risk diagnostics.**

SignalForge-QX studies whether public macroeconomic, energy-inventory, positioning, and fund disclosures improve weekly multi-asset forecasts when publication timing, revisions, missingness, and model selection are treated explicitly. This publication contains the original research code and the isolated SIA, BAR, provenance, OOF, and model-risk extensions through **v3.8**.

> **Research status:** The stated historical development and software-diagnostic scopes are complete. The system is **not** a qualified trading strategy. Authenticated original-publication Tier-A clocks, full P1 economic data, independent scientific freeze, sealed final evaluation, and genuine future forecast validation remain unqualified.

## Predictive development results

Primary score: **date-equal, asset-equal, seed-averaged normalized pinball loss (NPL)** across five quantiles. Lower is better.

| Historical development track | Original RGMF | Strong statistical I0 | SIA v2 | BAR | Calibrated SIA |
|---|---:|---:|---:|---:|---:|
| Main-A, 2020 | 0.500614 | 0.498910 | **0.493252** | 0.500807 | 0.494581 |
| Nested-B, 2022 | 0.322948 | **0.272733** | 0.274078 | 0.274928 | 0.275097 |

On Main-A, SIA reduced NPL by **1.134% versus the strongest statistical comparator** and by **1.47% versus the original RGMF**. On Nested-B, SIA reduced NPL **15.13% versus the original RGMF**, but performed **0.493% worse than the strongest statistical comparator**. These are previously observed, source-clock-unqualified (Tier B) *development* experiments, not independently confirmed future performance. BAR and additional quantile calibration did not satisfy promotion criteria.

Eight ETF instruments: **SPY, QQQ, IEF, TLT, GLD, SLV, USO, UNG**. Each matched model/track comparison includes 52 distinct market dates, eight assets, three saved training seeds, and the same prediction grid. Seeds and assets do not increase the number of independent market dates.

## Model-risk findings

- **Tail calibration risk:** Nominal 90% SIA forecast intervals had 79.97% (Main-A) and 78.77% (Nested-B) empirical coverage on the development cohorts.
- **Coverage transfer:** Genuine mature historical SIA OOF 90% coverage was 91.43%/93.18%; later development coverage was 79.97%/78.77%, respectively. These are *different calendar cohorts*, not a causal effect estimate.
- **Failed calibrator:** Train-only calibration raised NPL on both tracks. Main-A deterioration concentrated in q05 and q95; Nested-B deterioration came from q50; unsupported Nested tails remained unchanged.
- **Stress and heterogeneity:** Main-A showed NPL improvement versus strongest I0 on six of eight assets, but no all-track dominance. Exploratory high-volatility slices improved on both tracks, without independent validation.
- **Monitoring:** 4,737 historical, maturity-aware **TEST_ONLY** threshold-replay records were computed. Historical exceedance frequencies are not controlled false-positive rates.
- **P0 financial materiality:** Hypothetical capped portfolio weight, concentration, and model-disagreement proxies were evaluated. No execution fills, net P&L, Sharpe, alpha, or actual economic return are claimed.

## Reproducibility and systems

- Admitted successor program: **540 fixed-setting GPU model/metric results**, including 530 canonical calculated results and ten eligible pilot reuses, and **2,849 validated CPU model/metric pairs**. Distinct result/fit/seed counts are not interchangeable.
- **120 genuine CUDA mature SIA OOF fits**, 12,168 OOF predictions, and **120/120 bitwise identical checkpoint-prediction replays** on the original qualified stack.
- **12,480 matched five-model development predictions** rechecked for grids, finite quantiles, and numerical scoring.
- Fixed MLP GPU profile: microbatch 32 took 12.725 s versus microbatch 128 taking 4.308 s, a **2.95x observed training-time difference on that single workload**; not a project-wide speedup.
- Intentional one-process CUDA restart at step 50, then resumed through 100 steps with exact parameter and prediction parity for that workload.
- Latest v3.8 evaluation: **98/98 scoped tests passing** (75 new v3.8 checks and 23 earlier pure diagnostic checks) and two independent numerical recomputations. This is not a whole-repository clean-room scientific qualification result.

## Code map

| Directory | Purpose |
|---|---|
| **src/signalforge/** | Source parsers, PIT selector, models, CUDA execution, scoring, runtimes and accounting contracts |
| **scripts/**, **tests/** | Public Python support scripts and model/contract tests |
| **configs/**, **contracts/** | Model/execution configurations and software acceptance-test catalog |
| **policy_v3/**, **studies_v3/** | Isolated research admission and baseline-anchored residual challenger (BAR) |
| **diagnostics_v3/** | Post-hoc errors, calibration, model-risk and challengers |
| **calibration_v34/** | Genuine mature SIA OOF and train-only calibration |
| **p1_pit_readiness_v34/**, **qualification_v35/** | Provenance, action/price eligibility, and future-contract gates |
| **data_recovery_v36/** | Bounded official documentary evidence and synthetic-only shadow recorder |
| **evidence_procurement_v37/** | Validated externally authorized dataset-import schema and action reconciliation |
| **model_risk_v38/** | Five-family probabilistic calibration, stress, chronological monitoring, seed/challenger, and P0 analysis |

## Evidence and public reproduction

The most complete report is [v3.8 model-risk and calibration analysis](evidence/reports/V38_INCREMENTAL_MODEL_RISK_AND_CALIBRATION_REPORT.md). See the [public evidence directory](evidence/) for selected technical reports, aggregate tables and three figures. See [reproducibility](docs/REPRODUCIBILITY.md) and [publication limitations](docs/PUBLICATION_BOUNDARY.md).

This is a **privacy-preserving publication snapshot**, not the original immutable research directory. Vendor or paid inputs, raw market prices/labels, protected source-event version histories, model checkpoint binaries, internal compute ledgers, credentials and all Codex/LLM prompts are intentionally absent. Some original hardcoded usernames are replaced with **USERNAME** in public code copies. Hence **public source files are not byte-identical to frozen internal execution code** and the published aggregates cannot be reproduced without the separately authorized original datasets, artifacts, dependencies, and paths.

The GitHub Actions workflow runs only dependency-free Python syntax and JSON checks; it does **not** run numerical research or download financial data.

### Scientific qualification state

| Boundary | Status |
|---|---|
| Retrospective scoped developer model and model-risk studies | Executed and reported |
| Authenticated first-public source-version clocks (Tier A) | **0 qualified** |
| Fully certified P1 economic asset sessions | **0 / 27,088** |
| Independent research/governance reviewer and actual freeze | Not obtained |
| Sealed final evaluation, real prospective forecasts and matured outcomes | **Not performed / 0** |
| Live strategy, actual alpha, Sharpe, net economic improvement | **Not claimed** |

### Repository license

No open-source license grant is included in this public snapshot. Contact the repository owner for reuse rights.

**Release line:** v3.8.0-research is a **pre-release research engineering milestone**, not an independently validated investment or trading system.
