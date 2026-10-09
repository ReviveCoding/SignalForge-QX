"""Verify unchanged research bundles/ledgers and checkpoint scoped continuation."""
import json
from collections import Counter
from signalforge.runtime import paths,file_hash,code_hash,now,digest,validate_bundle,commit_bundle,atomic_json
from signalforge.successor import ledger_snapshot
from signalforge.completion import current_full_validation
repo,runtime=paths();reports=repo/'reports';validation=current_full_validation(repo)
if not validation['passed']:raise PermissionError('Current full source-bound validation required')
parent=ledger_snapshot(runtime/'ledger/development_compute.sqlite',43200)
if parent['entries']!=3610 or parent['charges_digest']!='2705b3d5e54f99b4d5813cdab40a3de9ab9dfe849315f46f95075c47953748cb':raise PermissionError('Parent ledger mutation')
full=ledger_snapshot(runtime/'ledger/successor_gpu_full_compute.sqlite',21600);pilot=ledger_snapshot(runtime/'ledger/successor_gpu_pilot_compute.sqlite',3600)
if full['entries']!=597 or abs(full['charged_seconds']-5234.51164490595)>1e-8:raise PermissionError('Successor full ledger changed')
cpu=Counter();gpu=Counter();cpu_families={'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}
for directory in (runtime/'artifacts/track_development').iterdir():
    path=directory/'metrics.json'
    if not path.exists():continue
    metric=json.loads(path.read_text())
    if metric.get('track') not in {'Main-A','Nested-B'} or metric.get('family') not in cpu_families:continue
    if metric.get('state')!='SUCCEEDED':raise PermissionError('CPU failed metric bundle')
    validate_bundle(directory);model=validate_bundle(runtime/metric['model_relative_bundle'])
    if digest(model)!=metric['model_receipt_id']:raise PermissionError('CPU model hash mismatch')
    cpu[metric['track']]+=1
full_receipt=json.loads((reports/'successor_gpu_full.json').read_text())
for track in full_receipt['tracks']:
    for metric in track['results']:
        validate_bundle(runtime/'artifacts/track_development'/metric['run_id'])
        if digest(validate_bundle(runtime/metric['model_relative_bundle']))!=metric['model_receipt_id']:raise PermissionError('GPU model hash mismatch')
        gpu[metric['track']]+=1
if dict(cpu)!={'Main-A':1925,'Nested-B':924} or dict(gpu)!={'Main-A':360,'Nested-B':180}:raise PermissionError('Bundle count changed')
names=['track_model_forensics.json','track_model_forensic_interpretation.json','original_archive_evidence_research.json','issuer_action_evidence_2023.json','forward_diagnostic_admission_v32.json','forensic_figures.json','test_execution.json','strict_pit_final_gate_audit.json']
summary={'state':'COMPLETED_FEASIBLE_FORENSIC_CONTINUATION_SCIENTIFIC_GATES_BLOCKED','created_at':now(),'source_tree_hash':code_hash(repo),'validation':validation,'cpu_bundles':dict(cpu),'gpu_bundles':dict(gpu),'all_original_model_metric_bundles_valid':True,'parent_ledger':parent,'pilot_ledger':pilot,'full_ledger':full,'parent_ledger_unchanged':True,'successor_ledgers_unchanged':True,'evidence_hashes':{n:file_hash(reports/n) for n in names},'phases':{'A':'COMPLETED_POSTHOC_IMMUTABLE_FORENSICS','B':'FROZEN_FUTURE_PLAN_NOT_ADMITTED_SCIENTIFIC_RESOURCE_GATES','C':'COMPLETED_BOUNDED_ARCHIVE_RESEARCH_PARTIAL_ACTION_EVIDENCE','D':'SOFTWARE_ACCEPTANCE_GAPS_CLOSED_RESEARCH_GATES_REMAIN','E':'COMPLETED_FRESH_FULL_VALIDATION_AND_REPORTING'},'remaining_blockers':['H2/H3 selection-frozen matched tuning/compute and frozen outages not executed; future protocol requires genuine original source clocks, operational freeze, future outcomes and admitted compute','E03-E08 all-track ablations/transport/SSL/scale/HPO remain separately unqualified; original exact primary HPO bill not admitted','Tier-A per-row original release versions/clocks/corrections unavailable or unqualified for current full source chain','P1 complete all-asset/all-year actions/ex-pay/splits/raw-open clocks/as-of cash benchmark and entitlement','Auxiliary 2023 EIA arithmetic inconsistency now explained by official correction notice, but frozen source contract still rejects it','Freeze and operational freeze gates fail; reserved data sealed and no prospective forecast issued'],'implementation_complete':False,'reserved_evaluation_complete':False,'strict_final_claims_permitted':False,'reserved_access':False,'new_research_fits_executed':0}
identity=digest(summary);target=runtime/'artifacts/forensic_continuation_checkpoint'/identity;commit_bundle(target,{'checkpoint.json':summary},{'reserved_access':False});summary['relative_bundle']=str(target.relative_to(runtime));atomic_json(reports/'forensic_continuation_checkpoint.json',summary)
print(json.dumps({k:summary[k] for k in ['state','source_tree_hash','cpu_bundles','gpu_bundles','all_original_model_metric_bundles_valid','parent_ledger_unchanged','successor_ledgers_unchanged','relative_bundle']},indent=2))
