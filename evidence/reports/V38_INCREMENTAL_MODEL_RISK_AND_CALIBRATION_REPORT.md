# SignalForge-QX v3.8 incremental model-risk and calibration audit



Claim boundary: **EXPLORATORY_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION**. Developer self-audit, not external independent validation, regulatory approval, or a new holdout. All 2020/2022 development labels were already seen.



## Execution and preservation

CPU calculations only: five saved model families, two tracks, **12,480 outer rows**; each track/family has52 market dates ×8 assets ×3 seed repeats. Genuine SIA OOF has12,168 rows:441 Main-A and66 Nested-B dates. Existing120 actual CUDA fit/model/checkpoint bundles and48 v34 reports were verified; zero new fits, calibrator fits, GPU computations or downloads. Twelve prior SIA/BAR candidate bundles also revalidated without inference.



Original357 files and protected dev577 files matched their prior hashes at start and independent check; v2 code hash remains `471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870`. Original compute ledgers were **not reopened**, explicitly prohibited by this task. Ledger preservation references the existing v37 proof; this audit does not claim a fresh ledger-content read. Historical105 pass/1 fail source-hash incident and all previous receipts remain untouched. New v38 orchestration failures (Windows relative separators and hardware-name/device-alias assumption) are retained in failure receipts and attempted code; no scientific data repairs or exclusions.



Definitions were saved to frozen_plan.json before new panels were loaded; plan hash `c4b57cac3117068b1a608511f6eb67f158f4e1fc802092a4248917da71a811a9`. Method/coverage matrix separates old completed21 CPU perturbations and99 tables from the new five-family/replay work. ABC is synthetic-only; full HPO/SSL/36/2392-fit programmes are not executed. No old model or outcome is overwritten.



## Scoring and interpretation

Pinball uses max(tau*(target-prediction),(tau-1)*(target-prediction)) in native units, divided by the exact train-only shared scale. Five quantiles averaged; seeds averaged within date/asset, assets equally within date, dates equally. Conditional slices use this same nested aggregation and at least13 distinct dates; smaller slices are UNDERPOWERED. Overall primary rows never filtered. Absolute/raw loss and all per-quantile errors are retained. Quantile reliability deviation is descriptive mean absolute coverage deviation, **not** classification ECE or a full CDF/PIT statistic. Five quantiles cannot identify PIT uniformity. At52 independent dates, expected5% tail support is2.6dates, not62.4independent asset/seed outcomes.



| track | model | loss | raw_loss | coverage80 | coverage90 | normalized_bias | max_abs_forecast | p99_abs_forecast | catastrophic_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main-A | BAR | 0.500807 | 0.0180152 | 0.69391 | 0.782051 | 0.454402 | 0.317982 | 0.314094 | 0 |
| Main-A | SIA_calibrated_v34 | 0.494581 | 0.0176078 | 0.705128 | 0.788462 | 0.354375 | 0.360672 | 0.287083 | 0 |
| Main-A | SIA_v2 | 0.493252 | 0.0175942 | 0.700321 | 0.799679 | 0.358672 | 0.368463 | 0.301954 | 0 |
| Main-A | original_RGMF | 0.500614 | 0.017847 | 0.677083 | 0.794071 | 0.414251 | 0.498678 | 0.339692 | 0 |
| Main-A | strong_I0 | 0.49891 | 0.0180781 | 0.706731 | 0.793269 | 0.435197 | 0.329073 | 0.329073 | 0 |
| Nested-B | BAR | 0.274928 | 0.013994 | 0.607372 | 0.763622 | -0.199222 | 1.05201 | 1.05051 | 0 |
| Nested-B | SIA_calibrated_v34 | 0.275097 | 0.00977679 | 0.589744 | 0.78766 | -0.376675 | 0.457787 | 0.301677 | 0 |
| Nested-B | SIA_v2 | 0.274078 | 0.0095896 | 0.589744 | 0.78766 | -0.319739 | 0.457787 | 0.301677 | 0 |
| Nested-B | original_RGMF | 0.322948 | 0.0114923 | 0.467949 | 0.696314 | -0.498847 | 0.747801 | 0.33892 | 0 |
| Nested-B | strong_I0 | 0.272733 | 0.0143963 | 0.617788 | 0.778846 | -0.190359 | 1.1076 | 1.1076 | 0 |



SIA minus strong baseline: Main-A -0.005658702 (-1.1342%); Nested-B +0.001344311 (+0.4929%). BAR is worse by0.3801% /0.8046%. Calibrated SIA is worse than frozen SIA by0.2694% /0.3721%. These are matched-fold descriptive results, not final claims. Neither lower training loss nor more complex architecture warrants promotion.



## Calibration transport and damage

The genuine v34 SIA OOF supersedes the older report’s missing-SIA-OOF statement. BAR-specific OOF remains unavailable. The existing calibrator was train-only; this audit does not refit it. Independent scalar replay of frozen offsets and supported-segment projection matches all saved calibrated predictions exactly. Nested unsupported tails remain4992exact unchanged elements.



| track | model | coverage_q05 | coverage_q10 | coverage_q50 | coverage_q90 | coverage_q95 | width90 | descriptive_mean_abs_quantile_coverage_deviation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main-A | BAR | 0.0857372 | 0.120994 | 0.443109 | 0.814904 | 0.867788 | 0.137815 | 0.0561859 |
| Main-A | SIA_calibrated_v34 | 0.0873397 | 0.120192 | 0.488782 | 0.825321 | 0.875801 | 0.126594 | 0.0435256 |
| Main-A | SIA_v2 | 0.0809295 | 0.122596 | 0.490385 | 0.822917 | 0.880609 | 0.134793 | 0.0419231 |
| Main-A | original_RGMF | 0.088141 | 0.128205 | 0.460737 | 0.805288 | 0.882212 | 0.135134 | 0.0536218 |
| Main-A | strong_I0 | 0.0793269 | 0.117788 | 0.454327 | 0.824519 | 0.872596 | 0.143851 | 0.0491346 |
| Nested-B | BAR | 0.154647 | 0.239583 | 0.532853 | 0.846955 | 0.918269 | 0.333542 | 0.0723718 |
| Nested-B | SIA_calibrated_v34 | 0.152244 | 0.264423 | 0.609776 | 0.854167 | 0.939904 | 0.149563 | 0.0864744 |
| Nested-B | SIA_v2 | 0.152244 | 0.264423 | 0.565705 | 0.854167 | 0.939904 | 0.149563 | 0.0776603 |
| Nested-B | original_RGMF | 0.241987 | 0.402244 | 0.638622 | 0.870192 | 0.938301 | 0.154733 | 0.134872 |
| Nested-B | strong_I0 | 0.144231 | 0.235577 | 0.564904 | 0.853365 | 0.923077 | 0.3496 | 0.0736538 |



| track | metric | OOF | outer | outer_minus_OOF | OOF_dates |
| --- | --- | --- | --- | --- | --- |
| Main-A | loss | 0.172327 | 0.493252 | 0.320924 | 441 |
| Main-A | coverage_q05 | 0.0386432 | 0.0809295 | 0.0422863 | 441 |
| Main-A | coverage_q10 | 0.0986395 | 0.122596 | 0.0239567 | 441 |
| Main-A | coverage_q50 | 0.497638 | 0.490385 | -0.00725333 | 441 |
| Main-A | coverage_q90 | 0.894558 | 0.822917 | -0.0716412 | 441 |
| Main-A | coverage_q95 | 0.952948 | 0.880609 | -0.0723389 | 441 |
| Main-A | coverage90 | 0.914305 | 0.799679 | -0.114625 | 441 |
| Main-A | normalized_abs_error | 0.645744 | 1.36442 | 0.718679 | 441 |
| Nested-B | loss | 0.156589 | 0.274078 | 0.117488 | 66 |
| Nested-B | coverage_q05 | 0.0328283 | 0.152244 | 0.119415 | 66 |
| Nested-B | coverage_q10 | 0.0959596 | 0.264423 | 0.168463 | 66 |
| Nested-B | coverage_q50 | 0.42298 | 0.565705 | 0.142725 | 66 |
| Nested-B | coverage_q90 | 0.904672 | 0.854167 | -0.0505051 | 66 |
| Nested-B | coverage_q95 | 0.964646 | 0.939904 | -0.0247426 | 66 |
| Nested-B | coverage90 | 0.931818 | 0.78766 | -0.144158 | 66 |
| Nested-B | normalized_abs_error | 0.59853 | 1.04579 | 0.447262 | 66 |



Main-A OOF q95 coverage95.2948% becomes88.0609% outer;90% interval coverage91.4305% becomes79.9679%. Nested q10 moves9.5960%→26.4423%; q50 moves42.2980%→56.5705%. Different cohorts and changing train-only normalizers limit causal comparison. These transport gaps are consistent with distribution/regime shift; they do not prove a coding defect.



Nested’s positive median offsets correct an under-covered OOF median, but outer median is already over-covered: calibrated coverage becomes60.9776%. All Nested calibration loss damage comes from q50; unsupported tail quantiles stay identity. Main q05/q95 loss increases dominate, while q90 improves. Narrower intervals are not automatically better: Main90% coverage drops79.9679%→78.8462%.



| track | value | within_slice_date_equal_delta | signed_loss_budget_contribution |
| --- | --- | --- | --- |
| Main-A | q05 | 0.00436558 | 0.000873115 |
| Main-A | q10 | 0.000405312 | 8.10624e-05 |
| Main-A | q50 | 5.27662e-05 | 1.05532e-05 |
| Main-A | q90 | -0.00195455 | -0.00039091 |
| Main-A | q95 | 0.00377561 | 0.000755121 |
| Nested-B | q05 | 0 | 0 |
| Nested-B | q10 | 0 | 0 |
| Nested-B | q50 | 0.00509907 | 0.00101981 |
| Nested-B | q90 | 0 | 0 |
| Nested-B | q95 | 0 | 0 |



## Signed loss budgets, concentration and stress

Loss-budget contributions are signed shares of the untouched total loss difference. Asset/date/seed/quantile/regime partitions each sum to the overall delta; **do not add different partition dimensions together**. Overall loss concentration is different from incremental model-effect attribution.



| track | model | USO_loss_share | top5_week_loss_share | dates |
| --- | --- | --- | --- | --- |
| Main-A | BAR | 0.476857 | 0.570431 | 52 |
| Main-A | SIA_calibrated_v34 | 0.480238 | 0.577929 | 52 |
| Main-A | SIA_v2 | 0.480855 | 0.577533 | 52 |
| Main-A | original_RGMF | 0.479531 | 0.571839 | 52 |
| Main-A | strong_I0 | 0.477876 | 0.570455 | 52 |
| Nested-B | BAR | 0.0359555 | 0.207682 | 52 |
| Nested-B | SIA_calibrated_v34 | 0.0110735 | 0.218719 | 52 |
| Nested-B | SIA_v2 | 0.010156 | 0.218655 | 52 |
| Nested-B | original_RGMF | 0.00931694 | 0.211652 | 52 |
| Nested-B | strong_I0 | 0.0390611 | 0.206811 | 52 |



| track | value | signed_loss_budget_contribution | within_slice_date_equal_delta |
| --- | --- | --- | --- |
| Main-A | GLD | -0.000150089 | -0.00120071 |
| Main-A | IEF | 0.000437526 | 0.00350021 |
| Main-A | QQQ | -0.00138076 | -0.0110461 |
| Main-A | SLV | 9.97066e-05 | 0.000797652 |
| Main-A | SPY | -0.00144513 | -0.011561 |
| Main-A | TLT | -0.000279333 | -0.00223466 |
| Main-A | UNG | -0.0017059 | -0.0136472 |
| Main-A | USO | -0.00123473 | -0.00987781 |
| Nested-B | GLD | 0.00120452 | 0.00963613 |
| Nested-B | IEF | -0.000740593 | -0.00592475 |
| Nested-B | QQQ | 0.00286446 | 0.0229157 |
| Nested-B | SLV | 0.00126946 | 0.0101556 |
| Nested-B | SPY | 0.0026493 | 0.0211944 |
| Nested-B | TLT | 0.000722034 | 0.00577627 |
| Nested-B | UNG | 0.00124488 | 0.00995903 |
| Nested-B | USO | -0.00786973 | -0.0629579 |



| track | operation | excluded | original_delta | counterfactual_delta |
| --- | --- | --- | --- | --- |
| Main-A | LEAVE_ONE_ASSET_OUT | USO | -0.0056587 | -0.00505597 |
| Main-A | LEAVE_ONE_STRONG_BASELINE_STRESS_WEEK_OUT | 2020-04-24 22:00:00+00:00 | -0.0056587 | -0.00451851 |
| Main-A | REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS | 1 | -0.0056587 | -0.00451851 |
| Main-A | REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS | 3 | -0.0056587 | -0.00486864 |
| Main-A | REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS | 5 | -0.0056587 | -0.00655219 |
| Nested-B | LEAVE_ONE_ASSET_OUT | USO | 0.00134431 | 0.0105303 |
| Nested-B | REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS | 1 | 0.00134431 | 0.0016578 |
| Nested-B | REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS | 3 | 0.00134431 | -0.000123242 |
| Nested-B | REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS | 5 | 0.00134431 | 0.000198915 |



Main USO accounts for about48% of total loss, but removing it only for influence leaves SIA-minus-baseline -0.005055972; this is not evidence to delete USO. Removing2020-04-24 leaves -0.004518505. Signed primary improvement is therefore not entirely driven by that asset or week. All original stress rows remain in every primary metric. Residual signed bias, lag1–4/8 autocorrelation,13/26-date full-support rolling errors/coverage and ex-ante heteroscedasticity associations are in residual files. Correlation is descriptive, not proof of serial independence or source causality.



## Fixed regime and challenger diagnostics

Frozen pre-outer volatility,1w/4w-return, source availability/OOD and source-age thresholds are reused. Outcome-magnitude slices are explicitly OUTCOME_CONDITIONAL and cannot be prospective triggers. Source-mask perturbations are copied from existing21CPU runs: not causal, not exact CUDA ablation; historic maximum discrepancy4.3479e-6is preserved. log_volume is absent from saved ex-ante panel and marked BLOCKED_SAVED_FEATURE_ABSENT. No mixed canonical data was read.



| track | model | value | distinct_dates | loss | coverage90 | status |
| --- | --- | --- | --- | --- | --- | --- |
| Main-A | BAR | HIGH | 52 | 0.625771 | 0.795513 | DESCRIPTIVE |
| Main-A | BAR | LOW | 46 | 0.176506 | 0.905797 | DESCRIPTIVE |
| Main-A | BAR | MID | 52 | 0.285033 | 0.769872 | DESCRIPTIVE |
| Main-A | SIA_calibrated_v34 | HIGH | 52 | 0.615961 | 0.798062 | DESCRIPTIVE |
| Main-A | SIA_calibrated_v34 | LOW | 46 | 0.176193 | 0.913043 | DESCRIPTIVE |
| Main-A | SIA_calibrated_v34 | MID | 52 | 0.282211 | 0.779167 | DESCRIPTIVE |
| Main-A | SIA_v2 | HIGH | 52 | 0.614732 | 0.8096 | DESCRIPTIVE |
| Main-A | SIA_v2 | LOW | 46 | 0.176774 | 0.916667 | DESCRIPTIVE |
| Main-A | SIA_v2 | MID | 52 | 0.28054 | 0.793269 | DESCRIPTIVE |
| Main-A | strong_I0 | HIGH | 52 | 0.624876 | 0.817949 | DESCRIPTIVE |
| Main-A | strong_I0 | LOW | 46 | 0.176355 | 0.916667 | DESCRIPTIVE |
| Main-A | strong_I0 | MID | 52 | 0.282153 | 0.769872 | DESCRIPTIVE |
| Nested-B | BAR | HIGH | 52 | 0.292911 | 0.717949 | DESCRIPTIVE |
| Nested-B | BAR | LOW | 40 | 0.507939 | 0.458333 | DESCRIPTIVE |
| Nested-B | BAR | MID | 52 | 0.239338 | 0.837179 | DESCRIPTIVE |
| Nested-B | SIA_calibrated_v34 | HIGH | 52 | 0.283523 | 0.730235 | DESCRIPTIVE |
| Nested-B | SIA_calibrated_v34 | LOW | 40 | 0.499575 | 0.529167 | DESCRIPTIVE |
| Nested-B | SIA_calibrated_v34 | MID | 52 | 0.248415 | 0.851068 | DESCRIPTIVE |
| Nested-B | SIA_v2 | HIGH | 52 | 0.282748 | 0.730235 | DESCRIPTIVE |
| Nested-B | SIA_v2 | LOW | 40 | 0.498816 | 0.529167 | DESCRIPTIVE |
| Nested-B | SIA_v2 | MID | 52 | 0.247328 | 0.851068 | DESCRIPTIVE |
| Nested-B | strong_I0 | HIGH | 52 | 0.28986 | 0.727564 | DESCRIPTIVE |
| Nested-B | strong_I0 | LOW | 40 | 0.50399 | 0.475 | DESCRIPTIVE |
| Nested-B | strong_I0 | MID | 52 | 0.237547 | 0.852244 | DESCRIPTIVE |



| track | candidate | control | corrects_loss | hurts_loss | control_wrong_candidate_right | control_right_candidate_wrong | sign_disagreement | near_zero_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main-A | BAR | strong_I0 | 0.645833 | 0.354167 | 0.0280449 | 0.0328526 | 0.0608974 | 0 |
| Main-A | SIA_calibrated_v34 | SIA_v2 | 0.553686 | 0.446314 | 0.0120192 | 0.0112179 | 0.0240385 | 0.00240385 |
| Main-A | SIA_v2 | strong_I0 | 0.590545 | 0.409455 | 0.0633013 | 0.0560897 | 0.121795 | 0 |
| Nested-B | BAR | strong_I0 | 0.524038 | 0.475962 | 0 | 0 | 0 | 0 |
| Nested-B | SIA_calibrated_v34 | SIA_v2 | 0.41266 | 0.58734 | 0.0440705 | 0.0432692 | 0.0873397 | 0.00641026 |
| Nested-B | SIA_v2 | strong_I0 | 0.52484 | 0.47516 | 0.0801282 | 0.0969551 | 0.177083 | 0 |



Challenger corrections/hurts are paired asset-date-seed descriptive fractions. Sign agreement is not profit; near-zero control is fixed |q50/scale|<0.001, not tuned from outcomes. Rank ties resolve descending q50 then lexical asset. All date/asset corrections and stress/regime links are retained, not selected for favorable findings.



## Functional seed stability

Checkpoint parameter cosine/norms from previous audit are reused as limited evidence; latent permutation makes naive tensor cosine unsuitable. New stability uses forecast span, q50 sign flips, interval-width dispersion and rank changes. Deterministic strong-baseline copies are duplicates, never additional independent samples.



| track | model | median_span_normalized | median_sign_flip | width90_seed_std_normalized | max_quantile_span_normalized |
| --- | --- | --- | --- | --- | --- |
| Main-A | BAR | 0.00945013 | 0.0673077 | 0.0126199 | 0.0163919 |
| Main-A | SIA_calibrated_v34 | 0.15101 | 0.139423 | 0.184669 | 0.353422 |
| Main-A | SIA_v2 | 0.146578 | 0.151442 | 0.157325 | 0.335962 |
| Main-A | original_RGMF | 0.199797 | 0.377404 | 0.22501 | 0.534672 |
| Main-A | strong_I0 | 0 | 0 | 4.72284e-17 | 0 |
| Nested-B | BAR | 0.00729886 | 0 | 0.00669852 | 0.0153182 |
| Nested-B | SIA_calibrated_v34 | 0.246372 | 0.182692 | 0.206925 | 0.463343 |
| Nested-B | SIA_v2 | 0.243769 | 0.360577 | 0.206925 | 0.463358 |
| Nested-B | original_RGMF | 0.468018 | 0.233173 | 0.178159 | 0.736074 |
| Nested-B | strong_I0 | 0 | 0 | 6.74382e-17 | 0 |



| track | model | top1_switch | top2_jaccard | spearman |
| --- | --- | --- | --- | --- |
| Main-A | BAR | 0 | 0.662393 | 0.94536 |
| Main-A | SIA_calibrated_v34 | 0.532051 | 0.397436 | 0.500458 |
| Main-A | SIA_v2 | 0.429487 | 0.423077 | 0.552656 |
| Main-A | original_RGMF | 0.564103 | 0.42094 | 0.458028 |
| Main-A | strong_I0 | 0 | 1 | 1 |
| Nested-B | BAR | 0 | 1 | 0.989927 |
| Nested-B | SIA_calibrated_v34 | 0.0897436 | 0.568376 | 0.691545 |
| Nested-B | SIA_v2 | 0.775641 | 0.288462 | 0.288919 |
| Nested-B | original_RGMF | 0.49359 | 0.472222 | 0.523657 |
| Nested-B | strong_I0 | 0 | 1 | 1 |



## New causal mature-OOF monitor replay

**4,737 chronological statistic/window/date records** were independently recomputed. A lagging value is available only at that date’s maximum label_available_at; prior dates must be strictly earlier decisions and fully mature by that computation time. Error is absolute q50 forecast error normalized by each existing train-only scale; asset/seed repetitions are averaged into one market-date statistic.90%miss fraction is1-interval coverage. Leading width uses earlier saved predictions and no labels; prediction lineage is retrospective prequential, not authenticated historic live issuance.



For window W=13/26 and lookback L=26/52, threshold=q95 of the latest L **past rolling-W statistics**, requiring L+W-1prior eligible dates. Current statistic uses latest W mature dates including current date. No outer data, new calibration or outcome-driven threshold selection is used. Latest data can trigger overlapping runs, so q95 does NOT imply5% false positives. Label maturity lag median is6.10417calendar days; no error warning is asserted at forecast time.



| track | statistic | lookback | window | eligible_dates | exceedance_dates | exceedance_fraction | maximum_run | first_feasible_decision | state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main-A | normalized_median_absolute_error | 26 | 13 | 403 | 72 | 0.17866 | 13 | 2012-04-06 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_median_absolute_error | 26 | 26 | 390 | 80 | 0.205128 | 18 | 2012-07-06 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_median_absolute_error | 52 | 13 | 377 | 48 | 0.127321 | 20 | 2012-10-05 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_median_absolute_error | 52 | 26 | 364 | 56 | 0.153846 | 18 | 2013-01-04 23:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | interval90_miss_fraction | 26 | 13 | 403 | 77 | 0.191067 | 17 | 2012-04-06 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | interval90_miss_fraction | 26 | 26 | 390 | 77 | 0.197436 | 18 | 2012-07-06 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | interval90_miss_fraction | 52 | 13 | 377 | 48 | 0.127321 | 17 | 2012-10-05 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | interval90_miss_fraction | 52 | 26 | 364 | 67 | 0.184066 | 19 | 2013-01-04 23:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_interval90_width | 26 | 13 | 403 | 106 | 0.263027 | 30 | 2012-04-06 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_interval90_width | 26 | 26 | 390 | 124 | 0.317949 | 25 | 2012-07-06 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_interval90_width | 52 | 13 | 377 | 59 | 0.156499 | 18 | 2012-10-05 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Main-A | normalized_interval90_width | 52 | 26 | 364 | 73 | 0.200549 | 26 | 2013-01-04 23:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_median_absolute_error | 26 | 13 | 28 | 3 | 0.107143 | 3 | 2021-06-18 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_median_absolute_error | 26 | 26 | 15 | 2 | 0.133333 | 2 | 2021-09-17 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_median_absolute_error | 52 | 13 | 2 | 0 | 0 | 0 | 2021-12-17 23:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_median_absolute_error | 52 | 26 | 0 | 0 | NA | NA | NA | UNDERPOWERED_NO_ELIGIBLE_DATE |
| Nested-B | interval90_miss_fraction | 26 | 13 | 28 | 5 | 0.178571 | 3 | 2021-06-18 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | interval90_miss_fraction | 26 | 26 | 15 | 3 | 0.2 | 2 | 2021-09-17 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | interval90_miss_fraction | 52 | 13 | 2 | 0 | 0 | 0 | 2021-12-17 23:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | interval90_miss_fraction | 52 | 26 | 0 | 0 | NA | NA | NA | UNDERPOWERED_NO_ELIGIBLE_DATE |
| Nested-B | normalized_interval90_width | 26 | 13 | 28 | 1 | 0.0357143 | 1 | 2021-06-18 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_interval90_width | 26 | 26 | 15 | 0 | 0 | 0 | 2021-09-17 22:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_interval90_width | 52 | 13 | 2 | 0 | 0 | 0 | 2021-12-17 23:00:00+00:00 | DRY_RUN_DESCRIPTIVE |
| Nested-B | normalized_interval90_width | 52 | 26 | 0 | 0 | NA | NA | NA | UNDERPOWERED_NO_ELIGIBLE_DATE |



Main median-error L26/W13:72/403dates exceeded (17.866%); interval-miss77/403 (19.107%). Nested corresponding error3/28 (10.714%) and interval-miss5/28 (17.857%). This is historical diagnostic incidence, not verified false-positive rate. Nested L52/W13 has only2eligible dates; L52/W26 has0 and is UNDERPOWERED. q99 expected extreme-tail support is0.66dates for Nested: UNRELIABLE_TAIL; Main4.41dates also insufficient for calibrated tail-risk claims.



Predeclared circular date-block4/8/13 bootstrap,2000draws, fixed randomness produces **descriptive** bounds (paired_date_block_bounds.csv, monitor_date_block_bounds.csv). Market dates—not repeated rows—are resampled. Overlapping rolling windows, regime dependence, short cohorts, known outcomes and many comparisons prevent confirmatory confidence or model promotion. Main paired bounds change with block choice; do not choose the favorable block.



| track | candidate | control | block | delta | lower_descriptive | upper_descriptive |
| --- | --- | --- | --- | --- | --- | --- |
| Main-A | SIA_v2 | strong_I0 | 4 | -0.0056587 | -0.0112268 | 0.000330185 |
| Main-A | SIA_v2 | strong_I0 | 8 | -0.0056587 | -0.0117727 | -0.000286573 |
| Main-A | SIA_v2 | strong_I0 | 13 | -0.0056587 | -0.0114792 | -0.000911904 |
| Main-A | SIA_calibrated_v34 | SIA_v2 | 4 | 0.00132894 | -8.97713e-05 | 0.00316017 |
| Main-A | SIA_calibrated_v34 | SIA_v2 | 8 | 0.00132894 | -0.000230607 | 0.00310705 |
| Main-A | SIA_calibrated_v34 | SIA_v2 | 13 | 0.00132894 | -0.000113254 | 0.00295768 |
| Main-A | BAR | strong_I0 | 4 | 0.00189644 | 0.000190371 | 0.00402045 |
| Main-A | BAR | strong_I0 | 8 | 0.00189644 | 7.06789e-05 | 0.00419509 |
| Main-A | BAR | strong_I0 | 13 | 0.00189644 | 8.10994e-05 | 0.00426877 |
| Nested-B | SIA_v2 | strong_I0 | 4 | 0.00134431 | -0.00846015 | 0.0111042 |
| Nested-B | SIA_v2 | strong_I0 | 8 | 0.00134431 | -0.00889789 | 0.0115175 |
| Nested-B | SIA_v2 | strong_I0 | 13 | 0.00134431 | -0.00777089 | 0.0119211 |
| Nested-B | SIA_calibrated_v34 | SIA_v2 | 4 | 0.00101981 | 8.96816e-05 | 0.00192082 |
| Nested-B | SIA_calibrated_v34 | SIA_v2 | 8 | 0.00101981 | 0.000214207 | 0.00182204 |
| Nested-B | SIA_calibrated_v34 | SIA_v2 | 13 | 0.00101981 | 0.000400183 | 0.00167292 |
| Nested-B | BAR | strong_I0 | 4 | 0.00219439 | 0.00132776 | 0.00303035 |
| Nested-B | BAR | strong_I0 | 8 | 0.00219439 | 0.00125815 | 0.00302623 |
| Nested-B | BAR | strong_I0 | 13 | 0.00219439 | 0.00139052 | 0.00293104 |



OOF source age/availability dependency clocks are absent: BLOCKED_UNVERIFIED_CLOCK, so these leading monitors were not fabricated. Outer-only1976registered source/asset/date age checks use prior train-only thresholds and existing v34 maximum-feature-dependency checks; they remain Tier B reconstructed dry runs. Baseline OOF warning references remain baseline-specific; they were not reused as BAR thresholds. BAR and original-RGMF model-specific OOF warning replay remains blocked. No alert notification or monitoring service was deployed.



## P0 materiality and controls

Fixed top1/top2/linear_rank signal mappings, asset cap0.25, commodity cap0.35, gross<=1 and cash remainder were reused. All29,952saved four-family weights match the new arithmetic. The same fixed rule extends to calibrated SIA;4,680group observations cover five families. HHI/effective holdings, seed/date stress exposure, L1changes vs strongest and turnover proxy are hypothetical P0 quantities—not economic returns or implementable fills.



| track | model | mapping | gross_signal_weight | HHI_gross_normalized | effective_holdings | commodity_weight | L1_vs_strong | L1_turnover_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main-A | BAR | linear_rank | 0.870655 | 0.159702 | 6.26228 | 0.35 | 0.359005 | 0.0283838 |
| Main-A | BAR | top1 | 0.25 | 1 | 1 | 0.25 | 0 | 0 |
| Main-A | BAR | top2 | 0.488462 | 0.5 | 2 | 0.257692 | 0.483974 | 0.0653595 |
| Main-A | SIA_calibrated_v34 | linear_rank | 0.819623 | 0.154866 | 6.46479 | 0.348647 | 0.338168 | 0.169939 |
| Main-A | SIA_calibrated_v34 | top1 | 0.25 | 1 | 1 | 0.219551 | 0.224359 | 0.173203 |
| Main-A | SIA_calibrated_v34 | top2 | 0.421154 | 0.5 | 2 | 0.286538 | 0.432372 | 0.283987 |
| Main-A | SIA_v2 | linear_rank | 0.81802 | 0.154827 | 6.46702 | 0.348469 | 0.335322 | 0.165488 |
| Main-A | SIA_v2 | top1 | 0.25 | 1 | 1 | 0.224359 | 0.198718 | 0.169935 |
| Main-A | SIA_v2 | top2 | 0.421154 | 0.5 | 2 | 0.284936 | 0.416667 | 0.256863 |
| Main-A | original_RGMF | linear_rank | 0.888925 | 0.156282 | 6.40445 | 0.333725 | 0.42252 | 0.178431 |
| Main-A | original_RGMF | top1 | 0.25 | 1 | 1 | 0.169872 | 0.25641 | 0.156863 |
| Main-A | original_RGMF | top2 | 0.452885 | 0.5 | 2 | 0.204487 | 0.526923 | 0.225163 |
| Main-A | strong_I0 | linear_rank | 0.655556 | 0.145275 | 6.88347 | 0.35 | 0 | 0 |
| Main-A | strong_I0 | top1 | 0.25 | 1 | 1 | 0.25 | 0 | 0 |
| Main-A | strong_I0 | top2 | 0.35 | 0.5 | 2 | 0.35 | 0 | 0 |
| Nested-B | BAR | linear_rank | 0.745299 | 0.154056 | 6.49148 | 0.35 | 0.106983 | 0.00244933 |
| Nested-B | BAR | top1 | 0.25 | 1 | 1 | 0.25 | 0 | 0 |
| Nested-B | BAR | top2 | 0.35 | 0.5 | 2 | 0.35 | 0 | 0 |
| Nested-B | SIA_calibrated_v34 | linear_rank | 0.767557 | 0.155276 | 6.45716 | 0.35 | 0.184599 | 0.0806033 |
| Nested-B | SIA_calibrated_v34 | top1 | 0.25 | 1 | 1 | 0.24359 | 0.0224359 | 0.0326797 |
| Nested-B | SIA_calibrated_v34 | top2 | 0.429808 | 0.5 | 2 | 0.29359 | 0.288462 | 0.101961 |
| Nested-B | SIA_v2 | linear_rank | 0.824003 | 0.15732 | 6.37269 | 0.349466 | 0.298487 | 0.140697 |
| Nested-B | SIA_v2 | top1 | 0.25 | 1 | 1 | 0.152244 | 0.320513 | 0.179739 |
| Nested-B | SIA_v2 | top2 | 0.446154 | 0.5 | 2 | 0.220192 | 0.477564 | 0.180392 |
| Nested-B | original_RGMF | linear_rank | 0.705057 | 0.146531 | 6.84447 | 0.35 | 0.227546 | 0.121258 |
| Nested-B | original_RGMF | top1 | 0.25 | 1 | 1 | 0.248397 | 0.320513 | 0.0653595 |
| Nested-B | original_RGMF | top2 | 0.375962 | 0.5 | 2 | 0.332692 | 0.290705 | 0.0862745 |
| Nested-B | strong_I0 | linear_rank | 0.683333 | 0.147975 | 6.75788 | 0.35 | 0 | 0 |
| Nested-B | strong_I0 | top1 | 0.25 | 1 | 1 | 0.25 | 0 | 0 |
| Nested-B | strong_I0 | top2 | 0.35 | 0.5 | 2 | 0.35 | 0 | 0 |



All21monitor definitions and11risk-register rows mapped to evidence, observed breach/support, source clocks, proposed UNAPPOINTED owner/cadence/escalation and hard-stop vs advisory controls. Current version is model_risk_control_map_current.csv; initial v38 map is retained. PSI/KS drift remains NOT_IMPLEMENTED_FORMAL_DISTRIBUTION_DRIFT, no after-the-fact threshold invented. Integrity guards are normative; statistical controls are unfrozen candidates. Synthetic injections test crossing, future feature, immature label, invalid/stale age, hash/checkpoint corruption, nonfinite values, duplicate key and absent signer. TEST_ONLY is not operational certification.



## Verification and limitations

Independent scalar branch and Decimal arithmetic reproduce all ten model/track scores and five quantile coverages. Maximum scalar-vs-Decimal row difference1.4211e-14. Separate scalar monitor reconstruction matches all4,737threshold/current values within2.7756e-15; exact frozen calibrator replay difference0. Current final scoped test receipt and additional independent verification are linked in final_completion.json. This is a developer two-implementation check, not external independent model validation.



BAR and calibrated SIA remain quarantined; frozen strongest I0 is retained as reference, SIA research-only. No architecture/seed/quantile/threshold/candidate was selected or promoted using these diagnostics. Genuine SIA OOF resolves one historic missing-data statement, but does not supply original publication evidence or a fresh market cohort.



Still blocked: authentic TierA clocks0; P1qualified0/27088; BAR-specific OOF; OOF source age clock lineage; appointed independent governance; valid operational/scientific freeze and independent future cohort (0dates). Reserved final stays sealed. No PnL/Sharpe/alpha/final/prospective claims permitted. Completion means this scoped CPU diagnostic audit, never whole-project scientific completion.



## Reproducible evidence

Code: model_risk_v38/source.py, diagnostics.py, monitor_replay.py, portfolio_proxy.py, supplement.py, controls_current.py, independent.py and scoped tests. Inputs: frozen source_index.json plus ext4 bundle hashes. Report/CSV/code output hashes and dedicated immutable ext4 archive are in final_completion.json. Run commands and failure traces are visible in the native Codex CLI; no unseen background delivery is promised.