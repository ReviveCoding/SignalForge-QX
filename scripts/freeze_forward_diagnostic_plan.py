"""Materialize a separate execution-ready plan, without allocating compute."""
import json
import numpy as np
from signalforge.runtime import paths,now,digest,file_hash,commit_bundle,atomic_json
from signalforge.forward_diagnostics import admission,planned_fit_counts
from signalforge.completion import current_full_validation
repo,runtime=paths();path=repo/'configs/forward_diagnostic_protocol_v32.json';plan=json.loads(path.read_text())
plan['exact_grid']={'lightgbm':np.geomspace(.0001,10000,20).tolist(),'rgmf_gru':[[w,lr] for w in [8,16,32,48,64] for lr in [.0003,.001,.003,.01]]}
plan['planned_fits']=planned_fit_counts(plan)
identity=digest(plan);target=runtime/'artifacts/forward_diagnostic_plan'/identity
commit_bundle(target,{'plan.json':plan},{'protocol_id':plan['protocol_id'],'no_training':True,'reserved_access':False})
evidence={'explicit_separate_run_authorization':True,'authorization_scope':'User 2026-10-07 continuation request authorizes technically feasible diagnostics, without new resource allocation or reserved/final access','current_full_validation':current_full_validation(repo)['passed'],'measured_pilot_bound_to_plan':False,'original_source_clock_qualified':False,'operational_freeze_verified':False,'new_cohort_unseen':True,'future_cohort_available':False,'use_reserved_data':False,'conservative_gpu_seconds':plan['planned_fits']['total']*300}
receipt={**admission(plan,evidence),'created_at':now(),'protocol_id':plan['protocol_id'],'frozen_plan_id':identity,'relative_bundle':str(target.relative_to(runtime)),'config_sha256':file_hash(path),'planned_fits':plan['planned_fits'],'reservation_bound_gpu_seconds':evidence['conservative_gpu_seconds'],'bound_kind':'worst-case registered fit reservations, not a pilot measurement; admission withheld','existing_ledgers_changed':False,'existing_results_changed':False,'reserved_access':False,'qualified_for_final':False,'evidence':evidence,'scope':'future protocol/code contracts materialized; specialized future runner and exact future input/fold/outage hashes cannot exist before operational freeze; no new fits launched'}
atomic_json(repo/'reports/forward_diagnostic_admission_v32.json',receipt);print(json.dumps(receipt,indent=2))
