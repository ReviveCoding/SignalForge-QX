# Version identifiers and research qualification

This repository intentionally has **two independent version namespaces**:

1. **Research publication tags** such as `v3.8.0-research` and
   `v3.8.1-research` identify public report and evidence snapshots.
   The latter fixed documentation completeness, not model training.
2. **Python package version** `3.0.0.dev1` appears in both
   `pyproject.toml` and `src/signalforge/__init__.py`. It identifies
   the retained v3 research implementation, not the latest research
   publication tag.

Changing the package version merely to mimic a public report tag could
misrepresent the provenance of the originally executed research code.
Both Python declarations must agree; CI checks this invariant.

The `main` branch can receive documentation, security-hygiene and public CI
improvements after a publication tag. A tag remains immutable. Neither a
package version nor a GitHub release implies Tier-A scientific qualification,
independent future-market replication, live P&L or production readiness.
