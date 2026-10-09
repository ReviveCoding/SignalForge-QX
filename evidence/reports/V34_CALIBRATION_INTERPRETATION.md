# V34 calibration interpretation and recovery record

This is an executed post-result exploratory P0/Tier-B study on already inspected 2020/2022 development folds. No candidate promotion, unseen confirmation, P1 economics or scientific-final claim is made.

120 genuine CUDA SIA OOF fits (Main 102, Nested 18), plus two real CUDA integration fits, completed. Main has 441 and Nested 66 independent mature OOF dates; 12,168 OOF rows. All 2,496 calibrated outer rows remain. The study meter records 1,407.565811592984 wall seconds during measured fit attempts, including the 35.132266465997-second resource interruption. This is observed elapsed time including checkpoint/serialization overhead, not CUDA kernel time; no numerical ceiling, bill gate or manual cost approval was used. Original ledgers were never charged.

| Track | Original RGMF | Strong I0 | Frozen SIA | Calibrated SIA | Change versus SIA |
|---|---:|---:|---:|---:|---:|
| Main-A 2020 I3 | 0.500613953 | 0.498910347 | 0.493251645 | 0.494580587 | +0.269425% loss |
| Nested-B 2022 I4 | 0.322948001 | 0.272733280 | 0.274077591 | 0.275097405 | +0.372090% loss |

Primary metric is date-equal, asset-equal, seed-averaged normalized pinball. Calibrated Main still improves strong I0 by 0.867843%; Nested deteriorates versus strong I0 by 0.866827%. These are exact matched-fold comparisons, not substituted aggregate study scores. Per-seed, asset, fixed ex-ante regime slices, five-quantile pinball and coverage, raw loss, 80/90 width and interval score, bad-week concentration, and paired 4/8/13-date block bootstrap are retained in evaluation.json and slices.csv. Seeds are repeated market observations, not independent statistical samples. Intervals are descriptive, not significance or future generalization evidence.

## Failed improvement hypothesis
Train-only intercept adjustments selected on 13 held inner OOF dates slightly improved inner loss (Main 0.139881735 -> 0.139194507; Nested 0.155058918 -> 0.154930913), but failed to improve outer SIA loss. Main central-90 coverage fell 0.799679 -> 0.788462 and native width shrank 0.134793 -> 0.126594; lower-tail q05 coverage moved 0.080929 -> 0.087340 and q95 0.880609 -> 0.875801, further from nominal. Nested unsupported tails stay exact identity: q10 coverage remains 0.264423. Median coverage moved 0.565705 -> 0.609776, further from nominal 0.5. No quantile crossings or forecasts above absolute 100 arose.

The outcome is consistent with the limitations of transferring pooled historical residual offsets across market regimes. It does not establish a unique causal defect. Main frozen SIA assigns about 48.09% of normalized loss mass to USO; date-block uncertainty and year-specific shocks constrain generalization. Nested's 66 dates provide only 3.3 expected smaller-tail dates at q05/q95, below the preregistered 20-date expected-tail floor; asset/seed pooling cannot manufacture power. No alternative candidate, CQR retry, tail relaxation, HPO, ABC or BAR expansion was chosen after these outcomes. BAR's prior negative gate remains unchanged.

## Recoveries and evidence
- Shared-device vLLM contention triggered a cooperative STOP at 74 fits. The interrupted attempt and STOP text/hash are retained. Three exclusive lease/VRAM observations established recovery; 74 successful bundles were reused, one interrupted identity retried under a distinct attempt ID. No unrelated process was killed.
- NumPy boolean metadata caused the first calibration digest to fail. Separate serialization v1r1 converts NumPy scalars to their exact native JSON counterparts; OOF hash and numerical rules unchanged, zero GPU refits. Regression tests passed; old source/protocol remains frozen.
- Final archive initially used path-qualified artifact names, which the immutable runtime rejects. A separate tested flat-name/file-map adapter restores archival packaging without changing any model or metric. The failed archive attempt is preserved.
- P1 adapter daily_rows/daily mapping and missing process-local library path were diagnosed, failure receipts retained and fixed without global environment changes.

120 second-pass actual CUDA reloads match saved predictions bitwise; checkpoints and full retention were verified. Independent scalar recomputation checks coverage, all pinball components, interval metrics, error/concentration statistics and 4/8/13-date bootstrap. Per-date monitoring replay checks all 312 date-seed groups/2496 rows against reconstructed feature clocks, with zero live notifications and no backdated forecast claim. Thresholds are OOF-only proposals, not operationally frozen.

Original 357 source copies, original source hash 7cdf374c83afd29e07c79e2ea8a0cc10ff2cad3637cee6ff01fa95ba9f7049bf, three old GPU ledgers, completed SIA/BAR/posthoc bundles and frozen v2 source hash 471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870 remain preserved.

## External/scientific gates
The separate source audit verified 67,448 canonical versions and their raw/derived dependencies but certified zero original first-public clocks. Raw opens exist (27,088 rows); issuer payment dates match 24 of 438 observed actions, leaving 414 without that partial issuer match, plus completeness, original announcement/open clocks, splits, entitlement and cash-account evidence missing. EIA Table4 revision evidence remains unresolved; Appendix E only authenticates propane/propylene corrections. P1 and Tier-A remain blocked.

One new future evaluation design is bound to immutable historical candidate model/prediction and calibration hashes, but is not an operational/scientific freeze or training authorization. Actual freeze is absent, future mature dates are zero, independent governance unavailable; earliest actual schedule is undefined. Two genuinely new untouched 52-date cohorts remain required. Existing final authorization does not substitute for missing verified freeze/strict PIT. Reserved final remains sealed, forecasts issued zero, implementation_complete and reserved_evaluation_complete false.

Acceptance: consult final_independent_risk_validation.json, STATUS.json and terminal_handoff.json for completed test counts, hashes and preserved failure records. Successful software/retrospective execution does not imply research-final qualification.