# v3.3 comprehensive post-hoc model-risk audit

Executed developer outcomes/monitoring/materiality analysis using immutable development predictions. No training, calibration fit, provider acquisition, execution prices, reserved access, model promotion, or final claims. This is not regulatory compliance or an external independent governance review.

Model-risk framing: [Federal Reserve SR26-2, April17 2026](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) supersedes SR11-7. Here conceptual soundness, outcomes analysis, ongoing monitoring, effective challenge and materiality are analytical categories only.

## Actual methods versus missing methods

| Method | Actual status | Boundary |
|---|---|---|
| PHT/semantic age inspiration | SIA implemented and6 real CUDA fits | Fixed log1p release/reference ages, explicit binary masks/constant-column OOD; full published-PHT reproduction not established |
| Baseline/physics-inspired residual | BAR implemented and6 real CUDA fits; negative vs strong controls | Statistical anchoring, not physical equations/PDE constraints; old v2 BAR gate unchanged |
| ABC asymmetric source correction | SYNTHETIC_PROTOTYPE_ONLY | No real fits or calibration, no tested investment alpha |
| Availability masks / SIA OOD | IMPLEMENTED_AND_EXECUTED | Typed raw-value validity; OOD detects train-constant/unseen numeric columns only, not comprehensive distributional OOD |
| Mature prequential baseline OOF | IMPLEMENTED_AND_EXECUTED | 441/66 residual OOF dates; no SIA/BAR model-specific mature OOF predictions |
| Zero heads / bounded residual / quantile monotonicity / fallback | IMPLEMENTED_TESTED_EXECUTED | CPU perturbation validates saved BAR; fallback mechanically tested, not a production qualification |
| Reliability/gating / source bias | LEARNED_GATES_EXECUTED; DIAGNOSTICS_NOW_EXECUTED | Gates are not calibrated source truth probabilities; source perturbation is not causal ablation/retraining |
| Quantile-tail calibration for SIA/BAR | NOT_FITTED_BLOCKED_OOF_UNAVAILABLE | Outer evaluation outputs cannot train calibrators; baseline OOF is not corrected model OOF |
| Parameter/seed stability / challenger checks | POSTHOC_EXECUTED | Same architecture seed cosine only; neuron permutation makes weight cosine non-identifiable |
| 36-fit / ABC / full HPO / SSL / v3.2 expansion | NOT_EXECUTED_IN_THIS_STUDY | Original completed experiments remain distinct; user stopped36-fit programme; BAR negative expansion rule maintained |
| P0 portfolio materiality | SIGNAL_PROXY_EXECUTED | Ranks/capped weight changes only; actual orders/fills/costs/PnL absent |
| Tier-A / P1 / reserved final / prospective | UNQUALIFIED_NOT_EXECUTED | No original-publication certification, economic execution qualification, final unsealing or future outcomes |

## Matched forecast reproduction

52 market dates ×8 assets ×3 seeds per track/family;1,248 rows per track and2,496 per model,9,984 across4 families. Same date/asset/seed/target/scale/fold/source identities. Seeds/assets are correlated repeats. No favorable exclusions. Normalized loss is date-equal, asset-equal, seed-averaged. Raw loss is separate.

| Track/model | NPL | Raw pinball | MAE | q05/q10/q50/q90/q95 coverage | 90% coverage | 90% raw width |
|---|---:|---:|---:|---|---:|---:|
| Main-A:2020:BAR | 0.50080679 | 0.01801523 | 0.046853 | 0.0857,0.1210,0.4431,0.8149,0.8678 | 0.7821 | 0.137815 |
| Main-A:2020:SIA_v2 | 0.49325164 | 0.01759421 | 0.046824 | 0.0809,0.1226,0.4904,0.8229,0.8806 | 0.7997 | 0.134793 |
| Main-A:2020:original_RGMF | 0.50061395 | 0.01784696 | 0.047948 | 0.0881,0.1282,0.4607,0.8053,0.8822 | 0.7941 | 0.135134 |
| Main-A:2020:strong_I0 | 0.49891035 | 0.01807813 | 0.046944 | 0.0793,0.1178,0.4543,0.8245,0.8726 | 0.7933 | 0.143851 |
| Nested-B:2022:BAR | 0.27492767 | 0.01399396 | 0.033722 | 0.1546,0.2396,0.5329,0.8470,0.9183 | 0.7636 | 0.333542 |
| Nested-B:2022:SIA_v2 | 0.27407759 | 0.00958960 | 0.034880 | 0.1522,0.2644,0.5657,0.8542,0.9399 | 0.7877 | 0.149563 |
| Nested-B:2022:original_RGMF | 0.32294800 | 0.01149233 | 0.038011 | 0.2420,0.4022,0.6386,0.8702,0.9383 | 0.6963 | 0.154733 |
| Nested-B:2022:strong_I0 | 0.27273328 | 0.01439631 | 0.034639 | 0.1442,0.2356,0.5649,0.8534,0.9231 | 0.7788 | 0.349600 |

## Calibration, concentration, regime and error findings

Date-block moving bootstrap4/8/13 weeks with2,000 draws covers primary loss, quantile coverage, interval coverage/width and normalized MAE. These intervals are descriptive; multiple slices are not a confirmatory hypothesis family and no significance/null claims are made. Quantile/asset/seed coverage deviations and pinball are in quantile_calibration.csv; interval scores, bias, p95/p99 loss and concentration in metrics.json.

Main-A:2020: BAR vs strongest loss change 0.3801%; vs SIA 1.5317%. BAR q0.95 deviation -8.221pp and q0.10 deviation 2.099pp. Numerical crossings/catastrophes0 does not establish tail calibration. Largest absolute forecast 0.317982.
BAR loss contribution USO 47.69%, UNG 2.37%; top5 market dates contribute 57.04% of date-equal loss. These assets/dates remain included.
Ex-ante historical20-session volatility, train-only quartile boundaries: HIGH loss 0.625771, 52dates, DESCRIPTIVE_DEVELOPMENT_ONLY; LOW loss 0.176506, 46dates, DESCRIPTIVE_DEVELOPMENT_ONLY; MID loss 0.285033, 52dates, DESCRIPTIVE_DEVELOPMENT_ONLY.

Nested-B:2022: BAR vs strongest loss change 0.8046%; vs SIA 0.3102%. BAR q0.95 deviation -3.173pp and q0.10 deviation 13.958pp. Numerical crossings/catastrophes0 does not establish tail calibration. Largest absolute forecast 1.052014.
BAR loss contribution USO 3.60%, UNG 17.24%; top5 market dates contribute 20.77% of date-equal loss. These assets/dates remain included.
Ex-ante historical20-session volatility, train-only quartile boundaries: HIGH loss 0.292911, 52dates, DESCRIPTIVE_DEVELOPMENT_ONLY; LOW loss 0.507939, 40dates, DESCRIPTIVE_DEVELOPMENT_ONLY; MID loss 0.239338, 52dates, DESCRIPTIVE_DEVELOPMENT_ONLY.

Regimes use exact registered return_1w, return_4w and realized_vol_20 raw features already available at decision time; bounds from pre-outer mature training only. Realized target-magnitude cohorts are OUTCOME_CONDITIONAL_NOT_PREDICTIVE, not market regime predictions. Quarter/half-year, source masks/ages, width reference bins and per-asset slices retained. Cohorts<13 dates are UNDERPOWERED. EIA source is USO activity, not UNG natural gas; rare masks and mapping are not arbitrary data omissions.

## Parameters, seeds and frozen-model source sensitivity

Six SIA and six BAR model states read on CPU; tensor/head/gate norms, parameter element counts and same-architecture seed cosines recorded. Neuron permutation means raw cosine is not identifiable functional stability. CPU-vs-saved-CUDA strict replay tolerance failed for some fits; declared atol2e-7/rtol2e-6 was not relaxed. CPU_backend_discrepancy.json preserves errors; CPU perturbations are descriptive backend-specific sensitivities, not qualified exact CUDA replay. All-source-missing fallback equals anchor bitwise. No fit or CUDA call.

Main-A masked-source minus full NPL: cftc -0.00098125, eia +0.00028096, macro -0.00124710. Positive means masking hurt, negative means masking helped. Frozen perturbation is association/sensitivity, not causal source value or retrained ablation.
BAR vs strongest: top1 switch 0.00%, top2 Jaccard 0.3419, mean Spearman 0.3516. Each rank uses only8 assets; ties and seed repetitions recorded.

Nested-B masked-source minus full NPL: cftc -0.00048644, eia +0.00029865, macro -0.00174130, nport -0.00035181. Positive means masking hurt, negative means masking helped. Frozen perturbation is association/sensitivity, not causal source value or retrained ablation.
BAR vs strongest: top1 switch 0.00%, top2 Jaccard 1.0000, mean Spearman 0.9414. Each rank uses only8 assets; ties and seed repetitions recorded.

## Challenger attribution and P0 materiality

All six model pairings have per-quantile native/normalized disagreement, sign flips, weekly Spearman/Kendall/ties, top1 switches/top2 Jaccard and date-block uncertainty. BAR comparisons contain row-level changes in median error and loss with source/volatility tags; largest disagreements and weekly errors are retained. A median sign is a P0 close-benchmark target sign, not an execution signal.

Three fixed transparent signal-weight mappings: top1, top2 and linear rank. Long-only hypothetical rank weights capped per asset0.25, commodity total0.35 (GLD/SLV/UNG/USO), gross<=1, remainder unallocated. Weight changes are signal turnover proxies, not actual trades, costs or profitability. No PnL/Sharpe/alpha or target-weight return calculation.
Main-A linear_rank: mean L1 rank-weight difference from strongest 0.359005.
Main-A top1: mean L1 rank-weight difference from strongest 0.000000.
Main-A top2: mean L1 rank-weight difference from strongest 0.483974.
Nested-B linear_rank: mean L1 rank-weight difference from strongest 0.106983.
Nested-B top1: mean L1 rank-weight difference from strongest 0.000000.
Nested-B top2: mean L1 rank-weight difference from strongest 0.000000.

## Train-only threshold candidates and monitoring

Release-age quantiles and baseline OOF error/width p95/p99 are computed on pre-outer mature training only, with cutoff/support/feature and OOF lineage. These are candidates, NOT operational/scientific freezes or controlled-FPR limits. All observed retrospective breaches are illustrative. Baseline OOF belongs to fixed historical / fixed linear-quantile alpha0.1, not SIA/BAR and not the old Nested alpha0.088586679 control. SIA/BAR/original corrected-model calibration/error thresholds: BLOCKED_OOF_UNAVAILABLE. No calibrator fitted.
Normative integrity hard stops: nonfinite, crossing, future feature, immature label, wrong schema/hash, reserved access. Source-age/availability/drift/disagreement/width warnings require named owner, support/cadence and escalation; proposed owners are not actual governance appointments. Routine review must examine mature performance and source changes; fallback itself requires validated baseline inputs. PSI/KS thresholds not calibrated here and no FPR claim. Full catalog/risk register is in monitoring.json and risk_register.csv.

## Verification, errors and genuine blockers

23 version-specific diagnostic tests executed; actual test log in tests.json. Scalar independent date→asset→seed→quantile recomputation reproduces vectorized loss; all9,984 rows retained. Frozen v2 code_hash remains471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870. Original357 protected files/three original ledgers, v2 pilot/full ledgers, BAR observation databases and immutable completion hashes verified. Diagnostic code separately hashed; no src/scripts or old report/config changes.
The earlier expanded historical suite remains105 pass1 fail: initial-v1 plan insists on its earlier source hash instead of completed-v2 source. Archived test/plan/receipt unchanged; no all-suite-green claim.
Remaining: mature corrected-model OOF for calibration, independent future replication/monitoring cohort, real independent governance reviewer, authenticated Tier-A clocks, qualified P1 raw opens/actions/ex/pay/splits/cash and final freeze/authorization. No model promotion follows this audit; BAR negative expansion decision preserved.

## Data dictionary and reproduction

panel.csv (ext4 immutable bundle): track/year/model/date/asset/seed, five native quantiles, native target, train-only scale, IDs, pinball/error/coverage/interval diagnostics, decision-known regimes/source masks/ages. CSVs in this report folder contain aggregate slices or perturbation results. quantile_calibration: deviation pp is empirical minus nominal. source_ablation: masked minus full loss, no refitting. signal_weights: capped P0 proxy only. thresholds: historical pre-outer candidates, not deployment authority. bootstrap intervals: market-date blocks only.
Run from isolated root: `pwsh -NoProfile -File diagnostics_v3/Invoke-ModelRisk.ps1 -Action tests` then `-Action run`. All completed diagnostic outputs bind hashes. Reproduction refuses inconsistent frozen inputs and never retrains models.
