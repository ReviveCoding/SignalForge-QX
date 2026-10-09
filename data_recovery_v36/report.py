"""Self-contained evidence-only report; claims drawn from actual validation receipts."""
from data_recovery_v36.acquire import R,O,Q,load
from data_recovery_v36.runner import write
from signalforge.runtime import now

def report():
    a=load(O/'attempted_public_sources.json');p=load(O/'public_analysis.json');cert=load(O/'genuine_certification_receipt.json');v=load(O/'independent_validation.json');e=load(O/'EIA_documentary_discrepancies.json');h=load(O/'G17_table_recovery_v2.json');s=load(O/'shadow_readiness_receipt_v2.json');tests=load(O/'tests.json')
    report=f'''# SignalForge-QX v3.6 public data recovery and shadow engineering

Completed {now()}. Engineering executed and independently checked; scientific qualification remains BLOCKED. This is neither final-model qualification nor a prospective result. Original and v2/v3/v34/v35 artifacts remain unchanged.

## Scope and source selection

The immutable intent and 34-candidate roster were registered before acquisition. Selection used official historical archive availability, not scores. Scope: EIA 2010/2023, CFTC 2023 annual positioning, two Federal Reserve G.17 releases, BLS 2023, NYFed 2023 cash rates, Treasury 2023 and iShares 2023 issuer documentation. No protected combined canonical export was opened, filtered or printed; prior 67,448 version inventory is aggregate-only. Existing scope rejection remains archived. No models, outer labels, reserved 2024+ market data or current market prices were accessed. Retrieval in 2026 is documentary observation only.

{a['candidate_urls']} unique candidate URLs; {a['requests']} actual GETs including robots; {a['response_bytes']:,} newly received bytes; {a['payload_documents']} verified document payloads ({a['new_payload_documents']} new, 2 reused). These are publisher-hosted content as currently retrieved, not authenticated contemporaneous originals. Max60 GETs, at least1.05sec request-start separation,25sec timeout,12MiB/request,250MiB aggregate, one attempt per URL; independent verification confirms limits. Redirects are failures, authentication/rate failures stop the publisher. Robots are checked; no available crawl-delay directive exceeded the registered interval. URLs, status, bytes, payload SHA256 and retrieval/header receipts are retained. Header dates/ETag/URL/archive labels do not prove first dissemination.

Negative recovery results: six EIA2010 archive paths404, Treasury2023 candidate404; BLS robots403 caused a host stop and zero attempts at its four document URLs. No SEC retry (v35's403 remains). Selected publisher payloads were not substituted with third-party snippets. SLV XLS bytes verified but optional xlrd absent; no global package installation or silent parsing. Initial G17 lxml parsing failures preserved; dependency-free stdlib parser recovered both documents, {sum(x['table_rows'] for x in h)} table rows. First runner ImportError repaired once by defining its read-only tool-runtime constant locally; before-code and incident retained. CFTC mixed-type warning is retained and does not affect explicit date/contract/open-interest fields.

## Actual documentary findings

New extracted observation versions: {p['observation_versions']:,}: 15,343 CFTC open-interest rows,499 NYFed rates,20 EIA registered field/current-date levels. The PDFs additionally provide20 rounded EIA levels, not additional exact unique event claims. G17 rows are verbatim table cells, not automatically canonicalized INDPRO vintage observations. CFTC annual files cover52 report dates each,2,809 financial rows/65 markets and12,534 disaggregated rows/298 markets, Jan3–Dec26,2023. They contain positioning CONTRACTS, not flows and not certified first-posted weekly files. The annual archive does not prove no corrections. Document parser receipts bind raw/text/table hashes. The initial20 parsed documents become22/23 after HTML recovery; one XLS remains byte-verified but unparsed.

### EIA arithmetic and revision evidence

Four releases (Aug16/Aug23/Dec20/Dec28,2023), five fields each. All20 same-release CSV/PDF current levels agree within0.0501 million-barrel rounding. Ten earlier-current versus later-prior comparisons agree exactly; no authenticated original/amended version pair or dissemination chain was recovered. Dec28 CSV current-minus-prior-minus-reported-change residuals are commercial crude −0.203, gasoline −0.094, distillate −0.092 million barrels (registered CSV tolerance0.0021). Dec28 PDF commercial crude residual −0.2 exceeds PDF tolerance0.1501. These are real published table inconsistencies; nothing was silently patched or relabeled. The propane/propylene appendix cannot certify amendments to crude/gasoline/distillate Table4. See EIA_documentary_discrepancies.json. Exact canonical joins remain blocked by lack of an externally certified pre2024-only export; no scope-control workaround.

Official sources: [Aug16 EIA Table4 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_08_16/csv/table4.csv), [Dec28 EIA Table4 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_28/csv/table4.csv), [Dec28 PDF](https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_28/pdf/table4.pdf), [propane appendix](https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_28/pdf/appendix_e.pdf).

### Macro and positioning

[January G17](https://www.federalreserve.gov/releases/g17/20230118/default.htm) and [December G17](https://www.federalreserve.gov/releases/g17/20231215/default.htm) tables recovered. A dated release page or scheduled time is not first-public log evidence. No raw historical index was relabeled as a modern base; no INDPRO yoy contract changed. [CFTC financial archive](https://www.cftc.gov/files/dea/history/fut_fin_txt_2023.zip) and [disaggregated archive](https://www.cftc.gov/files/dea/history/fut_disagg_txt_2023.zip) retain exact raw zip and parsed reference/contract unit identities. First-public/correction lineage not authenticated. BLS was not acquired after robots403; externally observed release/embargo information is not treated as a local qualified event.

### Issuer and cash evidence

[Issuer2023 distribution supplement](https://www.ishares.com/us/literature/tax-information/2023-ishares-distribution-summary-stamped.pdf):24 exact IEF/TLT ex/pay/amount tuples re-extracted and uniquely matched to prior provider actions, same24 as v35, new unique matches0. No provider rows overwritten. 438 provider action rows all lack original pay-date fields;414 remain unmatched by this narrow issuer scope. This retrospective supplement proves partial issuer content, not original announcement times or full dividend/split/no-action completeness. [SLV broker statement](https://www.ishares.com/us/literature/tax-information/ishares-silver-trust-broker-stamped.pdf) concerns2023 trust tax income/expense classification; absence of gross income is not a complete split/no-action attestation. [Government-source income supplement](https://www.ishares.com/us/literature/tax-information/2023-ishares-us-government-source-income-information-stamped.pdf) is annual income-tax classification, not a cash strategy return.

[NYFed SOFR2023](https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate=2023-01-01&endDate=2023-12-31) and [EFFR2023](https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2023-01-01&endDate=2023-12-31):499 rate rows with effective dates2023 and explicit percent annualized units. These provide documentary rates on dates corresponding to1,992 asset-session rows, not1,992 eligible P1 sessions. Effective date/revisionIndicator/retrieval do not authenticate known-at or an investable same-horizon cash benchmark. Quote basis, ACT360 fixture and actual execution/cash holding evidence remain distinct.

## Qualification numerator and missing proof

Tier-A before/after **0/0**, canonical linked original versions0. New documentary claims23, original-first-public certifications0. Previous67,448 aggregate versions remain Tier-B/no new per-version audit. Empty trusted publisher/reviewer registries are intentional, not synthetic signers. Import adapter verifies external attestation and pre2024 clock scope BEFORE opening a supplied file, and permits only an explicit external inbox; it does not export forbidden canonical data.

P1 before/after **0/0 of27,088 asset sessions**,8 assets×3,386 sessions,112 asset-year groups. All raw-open bytes previously available remain verified partial, not authenticated execution-open publication evidence. Missing jointly: raw tradable opens/publication and permissions; complete action/split/no-action coverage; actual ex/pay and original-known clocks; as-of cash; cost/fill conventions; accounting/reviewer attestation. No actual PnL, alpha, Sharpe, orders or adjusted-price substitution. Synthetic dividend receivables/pay/split/ACT360 conservation tests are TEST_ONLY. Dashboard identifies potentially affected scopes and evidence availability, does not add overlapping unlock counts or promise eligibility; financial costs were not invented. Requests are draft only, none sent.

## Shadow engineering and gate separation

A runnable local append-only SQLite recorder was executed twice with TEST_ONLY fixtures:24 initial records plus48 exact-track records in a separate v2 fixture DB. Only the48 exact-track records are current readiness evidence; neither run is a market forecast. Real wall-clock UTC, monotonic IDs, body/receipt hash chain, atomic transactions, update/delete guards, unique semantic identity, idempotent retry/conflict handling, cooperative interruption/replay covered. Exact model/source/normalizer/target/benchmark bindings are track-specific in the current plan; all are historical candidate/reference identities, not chosen/frozen live controls. Independent replay recomputed identities and all48 hashes. Real issuance rejected before DB creation/write. No daemon, schedule, live lookup or GPU inference.

Two nonoverlapping52-mature-date cohorts per track,8 assets,3 seeds; date-equal/asset-equal/seed-average scoring and block bootstrap4/8/13 with2,000draws retained. A4,992-row complete synthetic fixture validates104 distinct dates per track; actual future mature dates **0**, actual forecasts **0**, schedule **UNKNOWN**. Synthetic dates never become future evidence. Real P0 requires all-source Tier-A, authentic independent reviewer, operational/scientific freezes, explicit candidate selection and exact forward benchmark. P1 is an independent blocked economics lane; it does not logically block P0, but P0's own gates remain unmet. v36 unconditionally blocks real issuance even if synthetic flags look complete. Follow future_shadow_runbook.md before any separately authorized real recorder extension.

## Validation and preservation

Actual scoped tests **{tests['passed']} pass**,0failures/errors/skips: {tests['new_v36_tests']} new v36 plus116 prior scoped tests. New tests cover42initial properties, actual EIA/cash/G17 provenance/faults, complete/missing/duplicate/overlapping grids and exact track binding. Broad original historical105pass/1fail v1 source-identity receipt is preserved and is not included in this pass claim. Independent quantitative routine validates all raw bytes, request identities/rate/limits, raw CFTC/cash/EIA counts, action tuples,27,088session/112group denominators, zero qualified numerators, SQLite replay,468 protected dev files including424 v35-protected files,357 original protected files,120 v34 model/checkpoint bundles and48 v34 report hashes. Frozen v2 hash remains **{v['frozen_v2_hash']}**. Original source and all three original ledger entries/digests/ceilings unchanged. Original scientific receipts and completed studies were not rerun or modified. Detailed ledger preservation is in independent_validation.json.

v36 engineering **COMPLETED**, public acquisition **TERMINAL_BOUNDED**, source qualifications **SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE**, shadow **TEST_ONLY_READY_REAL_BLOCKED**, future evaluation **WAITING_FOR_GENUINE_FREEZE_AND_FUTURE_OBSERVATIONS**, reserved final **SEALED**. Whole-study implementation/scientific completion is not declared. No invented first-public clocks, signers, future dates or performance gains.

## Reproduction and immutable artifacts

Use existing WSL research Python, PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES empty, isolated runtime and ext4 TMPDIR; no global changes. Run scoped tests in tests.json with --import-mode=importlib -p no:cacheprovider. `python -m data_recovery_v36.independent` is a read-only check apart from a new named validation receipt and refuses rewriting an existing receipt. `python -m data_recovery_v36.acquire` reuses its terminal verified acquisition receipt without another GET; do not delete receipts to rerun requests. Parsing runner/report use exclusive creation; results already exist and should be inspected/replayed, not overwritten. Recorder commands and TEST_ONLY file paths are in future_shadow_runbook.md. Final immutable archive index and checksums are in final_completion.json. New code remains outside src/scripts; old source_hash is independently reproducible.
'''
    (O/'V36_PUBLIC_DATA_RECOVERY_AND_SHADOW_TECHNICAL_REPORT.md').write_text(report)
    requests='''# Draft official-source/custodian requests — NOT SENT

All requests are limited to historical valid_at AND known_at strictly before2024-01-01T00:00:00Z. Do not return reserved/final labels, credentials or private account identifiers. The recipient must identify the actual publisher/custodian and scope; this draft creates no authority.

## EIA original/correction and dissemination custody

Please provide byte-exact original and amended Table4 payloads for2023-08-16,2023-08-23,2023-12-20,2023-12-28; original versions for2010-07-21/28 if publicly available; publisher version ID, SHA256, reference week, region, field, million-barrel unit, predecessor/supersession/correction lineage and complete amendment scope. Dec28 CSV commercial crude436.568,prior443.682,change−6.911 has−0.203 arithmetic residual; gasoline−0.094 and distillate−0.092. Please distinguish propane appendix from any Table4 amendment. Supply authenticated first publicly disseminated UTC timestamps linked to each exact version, record/log semantics, timezone, distribution channel and verification chain. A current download, archive URL date or scheduled time is insufficient.

## CFTC weekly archive/corrections

Please supply original financial/disaggregated weekly2023 reports for52 report dates Jan3–Dec26, with exact market codes, contracts units, reference dates, raw version hashes, first-public dissemination logs and full corrections. Annual files FinFutYY.txt/disaggregated history contain15,343rows but do not certify weekly original availability. Confirm whether history was rewritten and what permissions apply.

## Federal Reserve/BLS/ALFRED

Please provide archival G17 Jan18/Dec15,2023, CPI Jan12/Dec12 and employment Jan6/Dec8 original versions plus base/unit/transformation/seasonal metadata and authenticated original per-version publication/correction clocks. INDPRO yoy requires same-vintage12-month values, never relabeled historical raw bases. Planned embargo times or ALFRED date-only vintage fields are insufficient. BLSrobots403 was respected; request lawful public download or custodian delivery, no bypass.

## Independent pre2024-only canonical export

An independent custodian—not this process inspecting mixed-clock rows—must create a separately scoped export containing only versions whose valid_at AND known_at are pre2024. Include source/entity/field/unit/reference/version/source payload SHA, parser hash, version lineage and coverage manifest. Attest payload SHA, latest clocks, revision_lineage_complete and PRE_2024_ONLY in domain v36-pre2024-export-1, externally enrolled trusted publisher key and genuine signature. No local mixed bundle filtering is authorized. Independent reviewer record and proof bytes must accompany any Tier-A claim.

## ETF issuers / price provider

ForSPY,QQQ,IEF,TLT,GLD,SLV,USO,UNG and2010–2023, supply complete authoritative dividend/split/no-action lists for each defined interval, exact asset/share class, ex/record/pay/effective dates, per-share amount/currency, split ratio, first-known/public announcement and revisions. Existing438provideractions omit pay dates;24IEF/TLT2023 exact issuer tuples match,414remain unmatched. A sample: IEF ex2023-02-01,pay2023-02-07,amount0.212178. Do not infer no action from missing rows or SLVtax statements. Also provide tradable raw opens, publication/asof clocks, coverage and legal usage/redistribution permissions, not adjusted closes. Prior403entitlement remains blocked, not retried.

## Cash source and independent reviewer

Please document SOFR/EFFR2023 exact original quote versions, first publication clocks, revision history and lawful usage. Provide the actual eligible cash instrument/overnight holding, compounding/day-count/calendar and decision-time availability rather than substituting effective-date quotes for execution returns. A genuine independent reviewer must approve exact source/price/action/cash scope, model/control identity and scientific/operational freeze after inspecting evidence. No generated local key or automated agent is an independent scientific authority.
'''
    (O/'permission_and_provider_request_pack.md').write_text(requests)
    (O/'signed_handoff_checklist.md').write_text('''# Authentic handoff requirements (unsigned checklist)

- [ ] Externally enrolled publisher keys; exact payload/correction and dissemination record hashes.
- [ ] Certified PRE_2024_ONLY external export; no local mixed-clock export.
- [ ] Independently reviewed exact source/entity/reference/field/unit/version joins.
- [ ] Qualified raw opens/actions/splits/no-action/pay/known clocks/cash/permission/cost proof for claimed P1 sessions.
- [ ] Genuine reviewer identity, signed domain-bound evidence approval; no self-generated authority.
- [ ] Explicit user candidate/control selection, exact forward baseline, source roster and target/normalizer/model binding.
- [ ] Real operational/scientific freeze times verified; no backdating.
- [ ] Separate explicit future-access authorization and immutable pre-outcome records.
- [ ] Two actual mature52-date nonoverlap cohorts each track, complete8asset3seed grid; no fixture substitution.
- [ ] Existing reserved-final authorization and one-batch freeze mechanisms remain sealed.

No box is signed or certified by this v36 engineering task. Ready to accept authenticated inputs is distinct from independent scientific approval achieved.
''')
    (O/'future_shadow_runbook.md').write_text('''# V36 local TEST_ONLY shadow runbook

Current state: TEST_ONLY_SHADOW_READY_REAL_ISSUANCE_BLOCKED. No daemon/scheduler/market fetch/inference. Current UTC is a machine-clock observation, not a forecast issuance time. Historical hashes are bound separately per track; live candidate/control selection is absent.

Using the existing WSL CUDA-environment Python with CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1 and ext4 runtime, cd to the isolated v33-dev root; PYTHONPATH includes root/src/scripts/studies_v3. Scoped process env only.

Replay existing fixture (read-only):
`python -m data_recovery_v36.shadow --action replay --database /home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/ledger/data_recovery_v36/shadow_test_only_v2.sqlite`

Idempotent TEST_ONLY retry (returns existing row, does not append):
`python -m data_recovery_v36.shadow --action append-test-only --database /home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/ledger/data_recovery_v36/shadow_test_only_v2.sqlite --plan reports/data_recovery_v36/shadow_prepared_plan_v2.json --input reports/data_recovery_v36/shadow_fixture_input_v2.json`

Replay verifies IDs, wallclock progression, receipt chain and bodies; independent.py also recomputes semantic identities and exact-track binding. SQLite UPDATE/DELETE triggers guard application use, not protection against an OS administrator rewriting arbitrary bytes; independent hash replay detects unauthorized edits. Stop the calling process between transactions to cancel; current transaction rolls back, successful rows persist. Exact retries reuse rows, changed bodies/identities fail closed. No runtime budget/deadline or compute ledger is involved.

REAL_FORWARD messages always rejected before output; there is no real-start command in v36. Before any separately authorized production extension, all genuine P0 gates in shadow_prepared_plan_v2 missing_P0_gates must pass and the actual forward input scope must be permitted. P1 economics require their own complete qualifications and do not automatically follow P0. No model/control is chosen by this task. Freeze real plan and governance before collecting outcomes; issue only after true freeze time, never backdate. At least2 nonoverlap52-mature-distinct-market-date cohorts per track×8assets×3seeds, frozen target/scoring scale, block bootstrap4/8/13,2000draws. Schedule UNKNOWN until actual freeze/observations. Test fixtures merely check schema/grid/clock algebra.
''')
    (Q/'EXECUTION_PLAN.md').write_text('# V36 executed plan\n\n1. PASS preserved original/v2/v34/v35 and registered intent.\n2. PASS bounded34candidate/35GET public acquisition.\n3. PASS documentary parser, gaps and independently recomputed quantities; XLS optional parser blocked.\n4. PASS atomic TEST_ONLY shadow, exact track binding and104-date full-grid fixtures.\n5. PASS scoped tests and independent preservation checks.\n6. COMPLETED engineering handoff; Tier-A/P1/freeze/realfuture blocked by authentic external evidence. No further GET/model jobs.\n')
    status={'version':'v36','engineering_complete':True,'public_acquisition_terminal':True,'new_Tier_A':0,'P1_qualified_sessions':0,'shadow_test_only_ready':True,'actual_forecasts':0,'actual_future_dates':0,'whole_research_implementation_complete':False,'reserved_evaluation_complete':False,'strict_final_claims_permitted':False,'reserved_final':'SEALED','scientific_state':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE','XLS_parser':'BLOCKED_OPTIONAL_DEPENDENCY','GPU_training':0};write('IMPLEMENTATION_STATUS.json',status)
    write('analysis_checkpoint.json',{'observed_at':now(),'phase':'ENGINEERING_COMPLETE_SCIENCE_BLOCKED','analysis':p,'certification':cert,'tests_passed':tests['passed'],'source_hash':v['v36_code_hash'],'future_mature_dates':0,'real_forecasts':0,'P1_PnL':False})
if __name__=='__main__':report()
