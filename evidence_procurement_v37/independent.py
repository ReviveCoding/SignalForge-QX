"""Independent v37 audit/report. Read-only old evidence; new receipts only."""
import json,csv,sqlite3,hashlib,base64,subprocess
from pathlib import Path
from signalforge.runtime import now,file_hash,code_hash,digest,validate_bundle
from diagnostics_v3.audit import preserve
from qualification_v35.core import canonical,sha,signed_record
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering');Q=R/'evidence_procurement_v37';O=R/'reports/evidence_procurement_v37'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,x):
    with (O/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def v37hash():return digest({str(p.relative_to(Q)):file_hash(p) for p in sorted(Q.rglob('*')) if p.suffix in ['.py','.ps1','.json']})

def receipt_matches(path,stdout):
    return load(path)=={k:v for k,v in stdout.items() if k!='receipt'}

def independent():
    pre=load(O/'preflight.json')
    for path,h in pre['protected_files'].items():assert file_hash(R/path)==h,path
    v36=load(R/'reports/data_recovery_v36/final_completion.json');folder=T/v36['relative_bundle'];validate_bundle(folder);mapping=load(folder/'index.json')['mapping']
    for flat,relative in mapping.items():assert file_hash(folder/flat)==v36['files'][relative]==file_hash(R/relative)
    assert len(mapping)==71 and len(pre['protected_files'])==540
    idx=load(R/'reports/calibration_v34/final_index.json');validate_bundle(T/idx['relative_bundle'])
    for relative,h in idx['files'].items():assert file_hash(R/relative)==h
    fits=load(R/'reports/calibration_v34/oof_completion.json')['results']
    for fit in fits:validate_bundle(T/fit['relative_bundle']);assert file_hash(fit['checkpoint_path'])==fit['checkpoint_hash']
    assert len(fits)==120;preservation=preserve()
    assert code_hash(R)=='471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870'
    contacts=load(O/'contact_registry.json');assert len(contacts['records'])==8 and all(x['status']=='DRAFT_NOT_SENT' for x in contacts['records']);text=(O/'DRAFT_REQUEST_MESSAGES.md').read_text();assert text.count('Status: **DRAFT_NOT_SENT**')==7
    assert contacts['external_messages_sent']==contacts['api_requests']==contacts['purchases']==0
    trust=load(Q/'trust_registry.json');assert trust['environment']=='REAL' and not trust['independently_enrolled'] and not any(trust[k] for k in ['custodians','publishers','reviewers'])
    demo=load(O/'executed_import_demo.json');assert demo['new_real_export_rows_received']==demo['new_real_matches']==demo['Tier_A']==demo['P1']==0
    for value in demo['accepted_cli']:assert value['state']=='ACCEPTED_TEST_ONLY' and value['test_not_evidence'] and value['qualified_Tier_A']==value['P1_qualified_sessions']==0 and not value['real_forward_allowed'];assert receipt_matches(value['receipt'],value)
    rejection=demo['real_rejection_cli'];assert rejection['reason']=='TRUST_NOT_INDEPENDENTLY_ENROLLED' and receipt_matches(rejection['receipt'],rejection)
    data=Path(demo['payload_path']).read_bytes();assert hashlib.sha256(data).hexdigest()==demo['fixture_payload_sha256'];manifest=load(demo['manifest_path']);registry=load(demo['registry_path']);public=registry['custodians'][manifest['key_id']]['public_key_hex'];assert signed_record(manifest,{manifest['key_id']:public},'v37-authorized-pre2024-export-1');assert manifest['body']['payload_bytes']==len(data) and manifest['body']['test_not_evidence'] is True
    with sqlite3.connect(Path(demo['database']).as_uri()+'?mode=ro',uri=True) as db:rows=db.execute('SELECT identity,payload_sha,manifest_sha,receipt FROM imports').fetchall()
    assert len(rows)==1;i,p,m,r=rows[0];record=json.loads(r);assert i==hashlib.sha256(canonical([record['provider'],record['dataset_id']])).hexdigest() and p==hashlib.sha256(data).hexdigest() and m==sha(canonical(manifest));assert record['test_not_evidence'] is True and record['qualified_Tier_A']==0
    legacy=list(csv.DictReader((R/'reports/p1_pit_v34/available_action_rows.csv').open()));issuer=list(csv.DictReader((R/'reports/data_recovery_v36/issuer_actions_delta.csv').open()));assert len(legacy)==438 and all(x['ex_date']<'2024-01-01' and not x['pay_date'] for x in legacy) and len(issuer)==24
    values=json.loads(data)['records'][0]['values'];g=[x for x in legacy if x['asset']==values['ticker'] and x['ex_date']==values['exDate'] and abs(float(x['divCash'])-values['distribution'])<1e-10];assert len(g)==1 and demo['reconciliation_fixture']['matched_legacy_rows']==1 and demo['reconciliation_fixture']['new_real_provider_matches']==0
    old=load(R/'reports/data_recovery_v36/genuine_certification_receipt.json');assert old['P1_denominator']==27088 and old['after_Tier_A']==old['new_P1_sessions']==0
    shadow=load(R/'reports/data_recovery_v36/shadow_readiness_receipt_v2.json')
    with sqlite3.connect(Path(shadow['database']).as_uri()+'?mode=ro',uri=True) as db:assert db.execute('SELECT COUNT(*) FROM records').fetchone()[0]==48
    tests=load(O/'tests_final.json');assert tests['exit_code']==0 and tests['passed']==241 and tests['new_v37']==60 and tests['prior_v36_and_earlier']==181 and tests['source_hash']==v37hash()
    wrapper=load(O/'powershell_wrapper_execution.json');assert wrapper['expected_fail_closed_exit_code']==2 and wrapper['observed_exit_code']==2 and 'TRUST_NOT_INDEPENDENTLY_ENROLLED' in wrapper['stdout']
    result={'observed_at':now(),'passed':True,'source_hash':v37hash(),'protected_dev_files':540,'v36_archive_files':71,'original_protected_files':357,'v34_OOF_model_checkpoint_bundles':120,'v34_report_files':len(idx['files']),'original_v2_v34_v35_v36_preserved':True,'frozen_v2_source_hash':code_hash(R),'original_source_hash':code_hash(Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX')),'preservation':preservation,'draft_messages':7,'contact_registry_entries':8,'external_messages_sent':0,'authorized_real_exports_received':0,'Tiingo_API_data_requests':0,'new_actual_provider_matches':0,'issuer_existing_partial_matches':24,'new_issuer_matches':0,'Tier_A':0,'P1':0,'P1_denominator':27088,'current_REAL_trusted_keys':0,'TEST_ONLY_immutable_import_records':1,'actual_forecasts':0,'GPU_training':0,'reserved_access':False,'no_protected_canonical_export_read':True,'tests':tests,'powershell_wrapper_fail_closed_test_passed':True,'historical_v1_source_mismatch_failure_preserved':True};write('independent_verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['preservation','tests']}),flush=True);return result

def report():
    v=load(O/'independent_verification.json');demo=load(O/'executed_import_demo.json')
    text=f'''# SignalForge-QX v3.7 executed evidence procurement and qualification-closure engineering

Executed {now()} in the existing native Codex CLI. Engineering and independent verification are complete. Genuine data procurement/scientific closure is BLOCKED_EXTERNAL_EVIDENCE, not complete. No email, form,FOIA,purchase,APIdata request, credential search,GPU training,current-market/2024+ data read, model retuning, forecast or freeze occurred. Existing scientific findings and negative results remain unchanged.

## Practical procurement achievement

Seven tailored English **DRAFT_NOT_SENT** messages prepared: Tiingo,EIA,CFTC,iShares,FRED,authorized external custodian and independent reviewer. Contact registry has8entries including the NYFed cash documentation route. Custodian/reviewer identities remain to be designated; no fictitious address or authority. These are actionable review/send drafts, not received provider evidence. Ranking uses evidence dependencies/available source routes and exact scope, no model accuracy or invented benefit/cost score. Full historical coverage and narrow2023pilots are distinguished; overlapping gate coverage is not summed into an unlock claim.

Tiingo's separate corporate-actions API is the concrete new avenue: [distributions documentation](https://www.tiingo.com/documentation/corporate-actions/dividends) includes paymentDate,recordDate,declarationDate and distributionFrequency, while [splits documentation](https://www.tiingo.com/documentation/corporate-actions/splits) defines splitTo/splitFrom,factor and active/cancelled status. Both contain beta/enterprise and EOD-entitlement wording; actual entitlement has NOT been verified. The request asks for explicit provider confirmation, initialIEF/TLT2023 and then only authorized8-asset2010–2023history, null-date completeness, stable identities, cancellation/revision chains and license. It does not assume sample endpoint filters prevent protected-data return; a provider-certified scoped export is required. Currency is required supplemental metadata, not falsely asserted a documentedAPIfield. Prior403is not retried. [Official Tiingo pricing contact](https://app.tiingo.com/pricing/) supplies sales@tiingo.com (official indexed page; direct app rendering sparse). No costs quoted, paid activation or account upgrades performed.

[EOD documentation](https://www.tiingo.com/documentation/end-of-day) distinguishes raw/adjusted fields and subsequent evening corrections. Current retrospective values and declared calendar dates cannot authenticate original first-public times; raw-open clock/correction/permission requirements remain.

[EIA official contact section](https://www.eia.gov/petroleum/supply/weekly/wpsr_ir-notice_06102026.php) verifies eiainfopetroleum@eia.gov; only contact/documentation was used, no linked2026test market sample followed. The request asks the exact4Table4releases2023, original/amended values and dissemination logs plus an explanation of crude/gasoline/distillate residuals−0.203/−0.094/−0.092millionbarrels. It does not treat the propane appendix as Table4repair.

[CFTC official portal contact](https://publicreporting.cftc.gov/stories/s/r4w3-av2u) supports marketreports@cftc.gov/publicreporting@cftc.gov, verified from official-domain indexed content because direct client-rendered page was empty. Request focuses52individual2023futures-only original reports, corrections and actual per-version publication;15,343annual archive positions are not first-public clock proof or cash flow. [iShares official form](https://www.ishares.com/us/about-us/contact-us) is verified; no archival-team email invented. Request seeks full2023IEF/TLT dividend/split/no-action coverage and original notices for24existing tuples. [FRED contact](https://fred.stlouisfed.org/help/about/contact-us/contact-us), [vintage-date specification](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html) and [observations specification](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) establish inquiry routing and revision-value semantics; database vintage dates do not prove first dissemination. Registered5series/INDPROsame-vintageYoY contract unchanged. NYFed cash custody requests use [official contacts](https://www.newyorkfed.org/contacts) and [API terms/documentation](https://markets.newyorkfed.org/static/docs/markets-api.html);499rate quotes are not qualified as-of cash execution.

## Executed private export acceptance bridge

New code only evidence_procurement_v37/,new outputs only reports/evidence_procurement_v37/. Manifest requires actual custodian role and separately independently enrolled key, exact provider/publisher, domain-bound Ed25519signature, PRE_2024_ONLYsnapshot/temporal extrema, exact bytes/SHA/rows/version roster, scopedassets/series, excluded counts, no opaque attachments and legal research/local/entitlement permissions. License2023cannot authorize2010–2023; unresolvedrights fail closed without buying access. No time/financial compute budget was introduced. Byte/row caps are inbox parser/resource safety, not GPU budget.

Trust/scope/license validation happens BEFORE dataset path resolution/read/hash. Real files must be in the isolated ext4 external inbox. No original mixed canonical archive is searched, filtered or exported. Default REALtrustregistry is EMPTY and cannot admit actual submissions. Fixture keys exist only in explicitlyTEST_ONLYtemporary/demo state, never REALregistry. Independently audited key enrollment remains a genuine governance trust root; supplying a self-declared signed package cannot establish trust. Schema is a routing specification; executable core is normative.

Safety limitation is explicit: a verified independent custodian attests exact bytes are scoped. Without reading a payload it is impossible to prove a malicious trusted custodian's undisclosed content; pre-read tests prove unsafe/unsigned/mixed manifests are rejected without reading. Post-admission syntheticsemantic tests detect inconsistent scoped content. No actual forbidden rows were inspected. Cross-year2024dates must be withheld with a completeness gap count; no implicit no-action certification.

Field schemas handle dividend declaration/ex/record/pay,null fields,currency/frequency,cancelled distributions; splitratio/status; explicit no-action intervals; cash effective vs publication/known/decision time and ACT360/actual-account proof; rawunadjustedopen/correction/permission; original sourceevidence with unchangedv35byte/clock/revision/reviewer qualification. Publisher/reviewer scope and independent key material are checked, including no reviewer sharing custodian/publisher key. Unknownunits, duplicate/ambiguousevents, wrongcurrentrevision, contradictions between complete no-action and actualactions fail closed. No historical original date field is overwritten. Tier-Aproof can only qualify after genuine evidence passes strict existing profile; P1jointsession qualification and scientific freeze are never inferred from an authorized export.

Real CLI execution: a signedTEST_ONLYone-rowdividendexport was accepted; immutable import registration persisted1row; exactretry returnedIDEMPOTENT_REUSE; a separate actualREALregistry call rejectedTRUST_NOT_INDEPENDENTLY_ENROLLED before opening the placeholderpayload. NativePowerShell7wrapper was also executed and returnedexpectedexit2with durable rejection. Receipts in import_receipts/ and executed_import_demo.json. Syntheticaction uniquely matches one safe historicalIEFexample and existing issuer amount/date; this is **0new actual provider matches**, not obtained data. Reconciliation covers issuerpaydateconflicts,cancellations and versions without mutating legacy438rows. Date-reached payment state is not actual transfer confirmation. Permaticker is retained but genuine issuer permanent-ID/share-class corroboration still required.

## Exact closure and blockers

| Evidence lane | Actual current result |
|---|---|
| Provider exports received / outbound requests sent |0 /0|
| First-disseminationTier-A before/after |0 /0|
| P1qualifiedasset-sessions |0 /27,088|
| Legacy provider actions missingpay fields |438,unchanged|
| ExistingIEF/TLTissuerpartialtuples /newunique |24 /0|
| TrustedREALcustodian/publisher/reviewer keys |0|
| Actual prospectiveforecasts/maturedates |0 /0|
| GPUfits /originalledgerchanges |0 /0|
| Reservedfinal |SEALED|

Exact missing next inputs are hard_blocked_requirements.json and seven drafts. Main practical next action: review drafts and separately authorize outbound contact; Tiingo/provider confirms actual entitlement and supplies licensedpre2024snapshotwith declaration/payment/split/fullcoverage; custodian and reviewer supply authenticidentity/proof enrollment and original dissemination/revisions; then explicitly invoke importer and inspect missing-proof receipts. No email was sent, provider response obtained, key enrolled or entitlement bypassed. P0 still independently needs TierA/reviewer/true scientific-operationalfreeze/usercandidate/exactbenchmark; P1addscompleteopens/actions/no-action/pay/cash/permissions/costfill/accounting. Independentfuture2×52maturedatecohorts unavailable, not fabricated. No genuineforward command in this task.

## Validation and preservation

Final scoped tests **{v['tests']['passed']}passed**,0failures/errors/skips: **60newv37 +181priorv36/earlier**. Initialnewtests50pass, strengthened56pass; firstfullscope240pass. Independentverification caught a CLIreceipt-path-only field comparison defect; failure/code preserved, comparison fixed and regression added. Final241tests include60newchecks; no scientific artifacts changed. Synthetictests are not research evidence. Historicaloldv1source-identity105pass/1fail receipt remains archived unchanged and is not represented as green.

Independentsecondpass verifies540protectedpriorfiles,v3671archived/currentbytehashes,original357files/threeledgers via preservationroutine,v34120OOFmodel/checkpointbundles and48reporthashes; verifies officialrouting/draftcounts, emptyREALregistry, fixturebytes/signature/immutableSQLiteidentity/retry/rejection, legacy438safeaction/24issuerreference counts, zeroactualqualification and wrapperexpectedgate. v2sourcehash **{v['frozen_v2_source_hash']}** unchanged. Newv37sourcehash **{v['source_hash']}** binds currentcode/PSwrapper/trustregistry; newreports are indexed in finalhandoff. No previousreport/schema/config/source/receipt mutation.

EngineeringstateCOMPLETED; evidenceclosureBLOCKED_EXTERNAL_PROVIDER_AND_INDEPENDENT_GOVERNANCE. Wholeprojectimplementation_complete=false,reserved_evaluation_complete=false,strictfinalclaims=false. User-facingdrafts/schema/runbook/tests/independentreceipt and immutablehandoff are saved locally. No backgroundexecutionpromise.

## Reproduction

Commands/paths in evidence_proof_submission_checklist.md. Use existingWSLPython, isolatedcwd/ext4TMPDIR, CUDA_VISIBLE_DEVICES empty, process-onlyPYTHONPATH/LD_LIBRARY_PATH. `python -m pytest evidence_procurement_v37/test_contract.py --import-mode=importlib -p no:cacheprovider -q` checksnewfixtures. Fullrequiredscopedfiles are in tests_final.json. `Validate-AuthorizedExport.ps1` is an explicit user-invoked bridge; exit0meansauthorizedpartialacceptance,notP1/final;exit2isrejection. Never rerun oldstudy/orchestration or delete receipts. `independent.py` writesnewexclusiveverificationreceipt and should be inspected or invoked with a separatelyversioned output on later delivery, not overwrite this completion. No public upload.
'''
    (O/'EXECUTED_EVIDENCE_PROCUREMENT_REPORT.md').write_text(text)
    write('IMPLEMENTATION_STATUS.json',{'version':'v37','engineering_complete':True,'evidence_procurement_closed':False,'state':'BLOCKED_EXTERNAL_PROVIDER_AND_INDEPENDENT_GOVERNANCE','draft_messages':7,'messages_sent':0,'real_provider_exports':0,'Tier_A':0,'P1_qualified':0,'actual_forecasts':0,'GPU_training':0,'reserved_evaluation_complete':False,'strict_final_claims_permitted':False,'whole_project_implementation_complete':False})
    write('progress.json',{'observed_at':now(),'stage':'V37_ENGINEERING_COMPLETE_EVIDENCE_BLOCKED','completed':['preservation','official_contact_documentation_verification','sevenDRAFT_NOT_SENTmessages','signedscopelicenseadmission','actionreconciliation','actualTEST_ONLYCLIretry/REALrejection','PowerShell7wrappertest','241scopedtests','independentverification'],'messages_sent':0,'real_provider_exports':0,'Tier_A':0,'P1':0,'actual_forecasts':0,'GPU_training':0})
if __name__=='__main__':independent();report()
