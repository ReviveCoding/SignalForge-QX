# SignalForge-QX v3.8.0-research

**Research pre-release. Engineering and retrospective diagnostics completed in stated scopes; scientific final qualification remains blocked.**

## Contents

- Core source-aware multi-asset probabilistic forecasting algorithms and SIA/BAR extensions.
- Mature OOF calibration code, publisher-data provenance gates, licensed-scope import validators, and maturity-aware monitoring diagnostics.
- Seven technical report narratives, five aggregated numerical tables and three model-risk figures.
- Privacy-preserving code snapshot excluding confidential raw data, internal model caches, credentials, ledger data, and all Codex/LLM prompts.
- Public CI that validates Python syntax and JSON structure only.

## Development results (lower normalized pinball loss is better)

| Historical development track | Strong I0 | SIA | BAR | Calibrated SIA |
|---|---:|---:|---:|---:|
| Main-A 2020 | 0.498910 | **0.493252** | 0.500807 | 0.494581 |
| Nested-B 2022 | **0.272733** | 0.274078 | 0.274928 | 0.275097 |

SIA improved NPL 1.134% versus the strongest baseline on Main-A, but underperformed that baseline by 0.493% on Nested-B. Nested-B SIA was 15.13% better than the older RGMF comparator, not 15.13% better than the strongest baseline.

Other measured results: 120 genuine SIA CUDA mature-OOF fits (12,168 OOF rows), 12,480 matched five-model development predictions, 4,737 historical monitor replay records, and 98 passing v3.8 scoped checks. No new training occurred during v3.8.

## Critical limits

Retrospective **Tier-B reconstructed-clock development** results are not independent future validation. New Tier-A certifications **0**; fully qualified P1 sessions **0/27,088**; prospective forecasts **0**. No Sharpe, alpha, real P&L, production trading, controlled monitoring false-positive rate or external independent governance approval is claimed.

Public source paths were privacy-redacted and may not match original frozen execution hashes. See [reproducibility](docs/REPRODUCIBILITY.md) and [claim boundaries](docs/PUBLICATION_BOUNDARY.md).
