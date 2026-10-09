"""Maintain truthful requirement mapping, completion vector and incremental cards."""
import json
from pathlib import Path
from signalforge.runtime import paths,atomic_json,now,file_hash,code_hash,commit_bundle,digest
from signalforge.cli import processed_audit

repo,runtime=paths();folder=repo/'reports'
def load(path):return json.loads((repo/path).read_text(encoding='utf-8-sig'))

mapping={
 'V3-T11':'tests/unit/test_forward_diagnostics.py::test_unseen_group_labels_cannot_fit_embedding_or_normalizer',
 'V3-T24':'tests/gpu/test_vectorized_oracle.py::test_vectorized_seeds_match_serial_rng_optimizer_and_independent_stopping',
 'P00-T03':'tests/unit/test_v3_core.py::test_admission_host_required',
 'P00-T08':'tests/unit/test_remaining_contracts.py::test_rootless_setup_has_no_global_settings_or_driver_commands',
 'P06-T05':'tests/real/test_workflow_evidence.py::test_real_all_information_and_capacity_baselines_share_grid_and_metadata',
 'P07-T03':'tests/real/test_workflow_evidence.py::test_real_nontrivial_baseline_search_has_complete_exact_recipe_proof',
 'P07-T06':'tests/real/test_workflow_evidence.py::test_real_null_targets_and_invalid_controls_are_retained_nonpromotable',
 'P10-T03':'tests/unit/test_experiment_integrity.py::test_hpo_never_outer_or_final',
 'P10-T08':'tests/unit/test_remaining_contracts.py::test_optimizer_failure_returns_registered_feasible_cash',
 'P11-T02':'tests/real/test_workflow_evidence.py::test_real_compile_includes_cold_cost_and_requires_numeric_parity',
 'P11-T03':'tests/real/test_workflow_evidence.py::test_real_microbatches_preserve_full_effective_optimizer_batch',
 'P11-T04':'tests/real/test_workflow_evidence.py::test_real_useful_counts_and_spawn_workers_do_not_inflate_sample_size',
 'P11-T05':'tests/real/test_workflow_evidence.py::test_real_useful_counts_and_spawn_workers_do_not_inflate_sample_size',
 'P12-T07':'tests/real/test_workflow_evidence.py::test_real_source_bug_preserves_originals_and_corrects_every_core_model',
 'P13-T08':'tests/unit/test_remaining_contracts.py::test_reporting_cannot_promote_capability_or_posthoc_slices_to_scientific_claims',
 'P14-T03':'tests/real/test_workflow_evidence.py::test_real_all_registered_results_and_negative_controls_are_retained',
 'P14-T04':'tests/real/test_workflow_evidence.py::test_real_posthoc_cases_remain_exploratory',
 'P14-T05':'tests/real/test_workflow_evidence.py::test_real_fresh_environment_reproduces_declared_cpu_evidence_tier',
 'P14-T07':'tests/unit/test_remaining_contracts.py::test_reporting_cannot_promote_capability_or_posthoc_slices_to_scientific_claims',
 'P15-T05':'tests/gpu/test_track_neural.py::test_training_raw_source_mask_exact_base_fallback_and_reload',
 'P15-T06':'tests/unit/test_remaining_contracts.py::test_forward_once_is_forecast_only_one_idempotent_iteration',
 'P15-T07':'tests/unit/test_statistical_interpretation.py::test_seeds_do_not_add_dates_and_underpowered_null_is_not_equivalence',
 'P15-T08':'tests/unit/test_remaining_contracts.py::test_forward_once_is_forecast_only_one_idempotent_iteration',
 'V3-T01':'tests/unit/test_remaining_contracts.py::test_native_launcher_keeps_interactive_workspace_auto_review_contract',
 'V3-T02':'tests/unit/test_remaining_contracts.py::test_native_launcher_keeps_interactive_workspace_auto_review_contract',
 'V3-T04':'tests/unit/test_remaining_contracts.py::test_resume_preserves_every_user_edited_desktop_input',
 'V3-T07':'tests/real/test_workflow_evidence.py::test_real_rootless_source_build_and_physical_host_disk_lineage',
 'V3-T08':'tests/unit/test_environment.py::test_inventory_wrong_gpu_cpu_opencl_or_objective_denied',
 'V3-T27':'tests/real/test_workflow_evidence.py::test_real_all_information_and_capacity_baselines_share_grid_and_metadata',
 'V3-T31':'tests/real/test_workflow_evidence.py::test_real_rootless_source_build_and_physical_host_disk_lineage',
 'P11-T06':'tests/gpu/test_checkpoint_adapter.py::test_actual_interrupted_optimizer_boundary_restores_rng_and_fullbatch_cursor',
 'V3-T30':'tests/unit/test_native_archive.py::test_native_node_checksum_mismatch_and_windows_archive_escape_denied',
 'P07-T05':'tests/real/test_public_development.py::test_real_complete_core_and_age_aware_baseline_receipts',
 'P01-T01':'tests/unit/test_desktop_study.py::test_track_targets_and_users_cannot_be_silently_collapsed',
 'P01-T02':'tests/unit/test_desktop_study.py::test_prior_method_claim_requires_reference_and_implementation_difference',
 'P01-T03':'tests/unit/test_desktop_study.py::test_positioning_cannot_be_labeled_cash_flow',
 'P01-T04':'tests/unit/test_desktop_study.py::test_access_policy_requires_every_source_license_auth_and_redistribution',
 'P01-T05':'tests/unit/test_desktop_study.py::test_mechanism_requires_competitor_and_falsification',
 'P01-T06':'tests/unit/test_desktop_study.py::test_dictionary_cannot_omit_units_entity_or_public_reference_clock',
 'P01-T07':'tests/unit/test_desktop_study.py::test_established_components_cannot_be_claimed_as_standalone_novelty',
 'P01-T08':'tests/unit/test_desktop_study.py::test_method_source_change_requires_registered_reason_and_adr',
 'P05-T07':'tests/unit/test_contamination.py::test_previously_inspected_dates_remain_consumed_after_split_rename',
 'P13-T07':'tests/unit/test_fill_costs.py::test_embedded_spread_is_not_deducted_again_and_infeasible_batch_is_atomic',
 'P13-T02':'tests/unit/test_statistical_interpretation.py::test_seeds_do_not_add_dates_and_underpowered_null_is_not_equivalence',
 'P13-T04':'tests/unit/test_statistical_interpretation.py::test_seeds_do_not_add_dates_and_underpowered_null_is_not_equivalence',
 'P14-T06':'tests/unit/test_reporting_cli.py::test_report_completion_dimensions_and_status_snapshot_lineage',
 'V3-T32':'tests/unit/test_reporting_cli.py::test_report_completion_dimensions_and_status_snapshot_lineage',
 'P03-T04':'tests/unit/test_cohorts.py::test_coverage_cohort_exit_missing_and_late_amendment_are_visible',
 'P14-T08':'tests/unit/test_research_cards.py::test_future_version_cannot_reuse_consumed_final_or_omit_failure_criterion',
 'V3-T29':'tests/unit/test_research_cards.py::test_future_version_cannot_reuse_consumed_final_or_omit_failure_criterion',
 'P12-T03':'tests/unit/test_final_execution_order.py::test_all_frozen_candidates_commit_before_labels_and_resume_immutable',
 'P12-T04':'tests/unit/test_final_execution_order.py::test_all_frozen_candidates_commit_before_labels_and_resume_immutable',
 'P12-T06':'tests/unit/test_final_execution_order.py::test_all_frozen_candidates_commit_before_labels_and_resume_immutable',
 'P00-T04':'tests/unit/test_v3_core.py::test_admission_host_required',
 'P02-T01':'tests/unit/test_acquisition_transport.py::test_partial_response_no_raw_receipt',
 'P02-T02':'tests/unit/test_v3_core.py::test_html_denial',
 'P02-T03':'tests/unit/test_v3_core.py::test_zip_paths',
 'P02-T04':'tests/unit/test_acquisition_transport.py::test_permission_denial_not_retried',
 'P02-T05':'tests/unit/test_v3_core.py::test_allowlist_and_redaction',
 'P02-T06':'tests/unit/test_acquisition_transport.py::test_partial_response_no_raw_receipt',
 'P02-T07':'tests/unit/test_acquisition_transport.py::test_changed_raw_bytes_immutable',
 'P04-T01':'tests/unit/test_starter_primitives.py::test_future_release_excluded',
 'P04-T02':'tests/unit/test_starter_primitives.py::test_old_revision_does_not_replace_new_reference',
 'P04-T03':'tests/unit/test_starter_primitives.py::test_future_revision_invariance',
 'P04-T04':'tests/unit/test_starter_primitives.py::test_source_ages_differ',
 'P04-T05':'tests/unit/test_data_targets.py::test_tier_a_rejects_unproven_claim',
 'P04-T06':'tests/unit/test_data_targets.py::test_mapping_valid_and_known',
 'P04-T07':'tests/unit/test_v3_core.py::test_nport_reference_not_zip_quarter',
 'P05-T01':'tests/unit/test_data_targets.py::test_calendar_holiday_dst',
 'P05-T02':'tests/unit/test_data_targets.py::test_p1_missing_stale_adjusted_reject',
 'P05-T03':'tests/unit/test_data_targets.py::test_p1_split_and_dividend_oracle',
 'P05-T04':'tests/unit/test_starter_primitives.py::test_label_publication_maturity',
 'P05-T05':'tests/unit/test_v3_core.py::test_actual_interval_purge',
 'P06-T01':'tests/unit/test_v3_core.py::test_train_only_transform',
 'P06-T02':'tests/unit/test_v3_core.py::test_train_only_transform',
 'P06-T07':'tests/unit/test_v3_core.py::test_train_only_transform',
 'P07-T01':'tests/unit/test_starter_primitives.py::test_pinball_median_absolute_half',
 'P07-T02':'tests/gpu/test_cuda_models.py::test_cuda_tree_objectives_and_reload',
 'P07-T04':'tests/unit/test_v3_core.py::test_statistical_save_parity',
 'P07-T07':'tests/unit/test_v3_core.py::test_missing_predictions_grid_preserved',
 'P07-T08':'tests/unit/test_v3_core.py::test_statistical_save_parity',
 'P08-T01':'tests/gpu/test_cuda_models.py::test_tiny_deterministic_recovery',
 'P08-T02':'tests/gpu/test_cuda_models.py::test_encoder_finite_order_save_causal',
 'P08-T03':'tests/gpu/test_cuda_models.py::test_rgfm_exact_fallback_and_masks',
 'P08-T04':'tests/gpu/test_cuda_models.py::test_rgfm_exact_fallback_and_masks',
 'P08-T05':'tests/gpu/test_cuda_models.py::test_rgfm_exact_fallback_and_masks',
 'P09-T02':'tests/unit/test_experiment_integrity.py::test_architecture_same_information',
 'P09-T03':'tests/unit/test_experiment_integrity.py::test_invalid_control_never_promoted',
 'P09-T04':'tests/unit/test_v3_core.py::test_outage_descendants',
 'P09-T08':'tests/unit/test_experiment_integrity.py::test_trial_failures_preserved',
 'P10-T06':'tests/unit/test_v3_core.py::test_calibration_support_and_ensemble',
 'P12-T01':'tests/unit/test_v3_core.py::test_final_denied_flags_without_evidence',
 'P12-T02':'tests/unit/test_experiment_integrity.py::test_same_frozen_batch_resume_only',
 'P12-T05':'tests/unit/test_experiment_integrity.py::test_same_frozen_batch_resume_only',
 'P12-T08':'tests/unit/test_v3_core.py::test_prospective_clock_replay_rejected',
 'P13-T01':'tests/unit/test_v3_core.py::test_paired_grid_mismatch',
 'P13-T03':'tests/unit/test_v3_core.py::test_holm_missing_family_member',
 'P13-T05':'tests/unit/test_v3_core.py::test_drifted_turnover_and_cost',
 'P13-T06':'tests/unit/test_v3_core.py::test_split_dividend_and_ex_entitlement',
 'P14-T01':'tests/unit/test_reporting_cli.py::test_incremental_report_blockers_and_lineage',
 'P14-T02':'tests/unit/test_reporting_cli.py::test_incremental_report_blockers_and_lineage',
 'P15-T04':'tests/unit/test_v3_core.py::test_prospective_clock_replay_rejected',
 'V3-T12':'tests/unit/test_v3_core.py::test_shared_normalizer',
 'V3-T13':'tests/unit/test_v3_core.py::test_missing_predictions_grid_preserved',
 'V3-T14':'tests/unit/test_v3_core.py::test_outage_descendants',
 'V3-T15':'tests/unit/test_v3_core.py::test_calibration_support_and_ensemble',
 'V3-T16':'tests/unit/test_v3_core.py::test_drifted_turnover_and_cost',
 'V3-T22':'tests/unit/test_v3_core.py::test_final_denied_flags_without_evidence',
 'V3-T23':'tests/unit/test_experiment_integrity.py::test_same_frozen_batch_resume_only',
 'P04-T08':'tests/unit/test_nport_parser.py::test_amendment_only_after_publication_and_fixed_cohort',
 'P10-T04':'tests/unit/test_ensemble.py::test_frozen_members_weights_and_separate_mean',
 'P10-T05':'tests/unit/test_ensemble.py::test_inverse_units_before_ensemble',
 'P11-T07':'tests/gpu/test_checkpoint_adapter.py::test_adapter_checkpoint_reuse_and_corruption',
 'P08-T06':'tests/unit/test_controller_contract.py::test_controller_mean_head_horizon_and_asof_cash',
 'P08-T07':'tests/unit/test_residual_oof.py::test_oof_future_target_feature_revision_invariance',
 'P09-T01':'tests/unit/test_registry.py::test_ablation_and_transport_types_separate',
 'P09-T05':'tests/unit/test_registry.py::test_ablation_and_transport_types_separate',
 'P09-T06':'tests/unit/test_registry.py::test_scale_grid_frozen_small_rule_and_train_weights',
 'P09-T07':'tests/unit/test_registry.py::test_ssl_rejects_future_lineage_before_cuda',
 'P10-T07':'tests/unit/test_controller_contract.py::test_controller_mean_head_horizon_and_asof_cash',
 'P00-T06':'tests/unit/test_gpu_lease_contract.py::test_external_cuda_owner_denied',
 'P03-T01':'tests/unit/test_starter_primitives.py::test_duplicate_version_rejected',
 'P03-T02':'tests/unit/test_eda.py::test_raw_qa_zero_missing_revision_and_order',
 'P03-T03':'tests/unit/test_eda.py::test_raw_qa_zero_missing_revision_and_order',
 'P03-T07':'tests/unit/test_eda.py::test_raw_qa_rejects_reserved_rows',
 'P03-T08':'tests/unit/test_eda.py::test_raw_qa_zero_missing_revision_and_order',
 'P03-T05':'tests/unit/test_eda.py::test_true_extremes_retained_and_conflicting_units_rejected',
 'P03-T06':'tests/unit/test_eda.py::test_true_extremes_retained_and_conflicting_units_rejected',
 'P04-T03':'tests/unit/test_panels.py::test_shared_feature_vectors_lineage_percentile_and_future_invariance',
 'P05-T06':'tests/unit/test_panels.py::test_grid_rejects_outcome_selection_naive_dates_and_reserved',
 'P05-T08':'tests/unit/test_track_engine.py::test_nested_cpu_full_two_folds_shared_information_artifacts_and_resume',
 'P06-T03':'tests/unit/test_panels.py::test_shared_feature_vectors_lineage_percentile_and_future_invariance',
 'P06-T04':'tests/unit/test_panels.py::test_context_masks_preserve_short_history_and_missing_distinctly',
 'P06-T06':'tests/unit/test_panels.py::test_shared_feature_vectors_lineage_percentile_and_future_invariance',
 'P06-T08':'tests/unit/test_input_identity.py::test_cache_data_identity_includes_parsed_values_clocks_and_contexts',
 'P08-T08':'tests/unit/test_experiment_integrity.py::test_hpo_never_outer_or_final',
 'P11-T01':'tests/unit/test_gpu_lease_contract.py::test_cross_framework_lease_excludes_second_owner',
 'P11-T08':'tests/unit/test_gpu_lease_contract.py::test_repeated_unexplained_shutdown_pauses_and_preserves_incidents',
 'V3-T09':'tests/unit/test_panels.py::test_nested_has_only_own_two_folds_and_mature_training',
 'V3-T10':'tests/unit/test_panels.py::test_grid_rejects_outcome_selection_naive_dates_and_reserved',
 'V3-T18':'tests/unit/test_stage_cache.py::test_reporting_postprocessing_do_not_invalidate_training',
 'V3-T20':'tests/unit/test_reporting_cli.py::test_incremental_report_blockers_and_lineage',
 'V3-T21':'tests/unit/test_experiment_integrity.py::test_invalid_control_never_promoted',
 'V3-T26':'tests/unit/test_experiment_integrity.py::test_measured_budget_bill',
 'V3-T28':'tests/unit/test_import_boundary.py::test_transitive_and_conditional_reserved_import_rejected',
 'P10-T01':'tests/unit/test_frozen_processing.py::test_frozen_calibration_bound_to_receipts_and_preserves_raw',
 'P10-T02':'tests/unit/test_frozen_processing.py::test_frozen_calibration_rejects_future_maturity_and_insufficient_tail_correction',
 'P15-T01':'tests/unit/test_forward_outcomes.py::test_mature_forward_outcome_append_only_grid_clocks_and_versions',
 'P15-T02':'tests/unit/test_forward_outcomes.py::test_mature_forward_outcome_append_only_grid_clocks_and_versions',
 'P15-T03':'tests/unit/test_forward_outcomes.py::test_mature_forward_outcome_append_only_grid_clocks_and_versions',
 'V3-T17':'tests/unit/test_auxiliary_clock_edge.py::test_same_friday_issue_is_not_future_target_from_date_upper_bound',
 'V3-T19':'tests/unit/test_experiment_integrity.py::test_retry_is_new_bounded_attempt_and_keeps_failure',
 'P00-T01':'tests/unit/test_environment.py::test_inventory_wrong_gpu_cpu_opencl_or_objective_denied',
 'P00-T02':'tests/unit/test_environment.py::test_inventory_wrong_gpu_cpu_opencl_or_objective_denied',
 'P00-T05':'tests/unit/test_environment.py::test_inventory_environment_drift_and_fixture_research_claim_denied',
 'P00-T07':'tests/unit/test_environment.py::test_inventory_environment_drift_and_fixture_research_claim_denied',
 'P02-T08':'tests/unit/test_environment.py::test_inventory_environment_drift_and_fixture_research_claim_denied',
 'V3-T03':'tests/unit/test_environment.py::test_inventory_wrong_gpu_cpu_opencl_or_objective_denied',
 'V3-T05':'tests/unit/test_environment.py::test_inventory_wrong_gpu_cpu_opencl_or_objective_denied',
 'V3-T06':'tests/unit/test_environment.py::test_inventory_wrong_gpu_cpu_opencl_or_objective_denied',
 'V3-T25':'tests/unit/test_environment.py::test_inventory_environment_drift_and_fixture_research_claim_denied',
}
# Verify exact test names against actual source, never mark a fabricated ID as implemented.
import ast
valid={}
for case,identity in mapping.items():
    path,name=identity.split('::')
    functions={n.name for n in ast.walk(ast.parse((repo/path).read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    if name in functions:valid[case]=identity
tests=load('reports/test_execution.json') if (folder/'test_execution.json').exists() else {'results':[]}
last_full=tests
if (folder/'cpu_test_execution.json').exists():
    cpu=load('reports/cpu_test_execution.json')
    if cpu['created_at']>tests.get('created_at',''):
        tests={**cpu,'results':cpu['results']+[r for r in tests['results'] if r['suite']=='gpu']}
suite_passed={r['suite']:r['exit_code']==0 for r in tests['results']}
executed={}
file_hashes={}
for result in tests['results']:
    for execution in result.get('executed_cases',[]):executed.setdefault(execution['identity'],[]).append(execution['state'])
    file_hashes.update(result.get('test_file_hashes',{}))
research_limitations={
 'V3-T11':{'state':'BLOCKED_DATA','reason':'Actual acquired physical panel has one entity; target-group zero-shot holdout is impossible. Group-fitted embeddings/normalizers are not used in its place.',
    'evidence':['reports/processed_data_audit.json','reports/auxiliary_all_family_temporal_transport.json','reports/track_input_qualification.json']},
 'V3-T24':{'state':'DEFERRED_METHOD','reason':'Optional vectorized independent-seed training was not adopted; all actual neural seeds used independent serial models and optimizers under one GPU lease. No fake multi-GPU/vectorized result.',
    'evidence':['reports/auxiliary_development_results.json','reports/real_training_systems_qualification.json']}}
blocked={key:value for key,value in research_limitations.items() if key not in valid}
cases=[]
for case in load('contracts/acceptance_tests.json')['cases']:
    identity=valid.get(case['id']);suite='gpu' if identity and '/gpu/' in identity else 'unit'
    actual_pass=bool(identity and executed.get(identity) and all(s=='PASSED' for s in executed[identity]) and
                     file_hashes.get(identity.split('::')[0])==file_hash(repo/identity.split('::')[0]))
    cases.append({'requirement_id':case['id'],'name':case['name'],'requirement_defined':True,
                  'test_implemented':bool(identity),'actual_test':identity,
                  'test_result':'PASSED' if actual_pass else 'NOT_RUN',
                  'qualification_scope':('real cached public development artifact check; strict all-track qualification remains separate' if '/real/' in identity else 'synthetic contract test; complete research execution gate remains separate') if identity else blocked.get(case['id'],{}).get('state','NOT_IMPLEMENTED'),
                  'requirement_complete_research_verified':False,
                  'reason':None if identity else blocked.get(case['id'],{}).get('reason','Additional implementation and requirement-specific test/evidence remain; not converted to PASS'),
                  'execution_evidence':'reports/test_execution.json' if identity else blocked.get(case['id'],{}).get('evidence'),
                  'blocked_evidence_hashes':{p:file_hash(repo/p) for p in blocked.get(case['id'],{}).get('evidence',[]) if (repo/p).is_file()}})
atomic_json(folder/'acceptance_mapping.json',{'cases':cases,'defined':160,'test_implemented':len(valid),
                                           'test_passed':sum(c['test_result']=='PASSED' for c in cases),
                                           'explicit_blocked_or_deferred':len(blocked),'all_defined_cases_accounted_for':all(c['test_implemented'] or c['qualification_scope'] in {'BLOCKED_DATA','DEFERRED_METHOD'} for c in cases),'complete_acceptance':False})
audit=processed_audit(repo)
aux=load('reports/auxiliary_development_results.json') if (folder/'auxiliary_development_results.json').exists() else {}
aux_success=[r for r in aux.get('results',[]) if r.get('state')=='SUCCEEDED']
if not aux.get('source_integrity_valid',True):aux_success=[]
eia=load('reports/eia_development_acquisition.json') if (folder/'eia_development_acquisition.json').exists() else {}
phases={}
from signalforge.track_status import track_phase_states
for phase in load('configs/phases.json')['phases']:
    state='PLANNED';reason='Implementation/integration remains executable and unfinished'
    if phase['id']=='P00':state='RUNNING';reason='Capability qualification passed; full resource/budget/recovery workflow qualification remains incomplete'
    if phase['id']=='P01':state='RUNNING';reason='Desktop definitions, literature attribution, source semantics and ADR contracts audited; external schemas and full scientific qualification remain separate'
    if phase['id'] in {'P02','P03','P04'}:state='RUNNING';reason='Real archives partially acquired and parsed; historical qualification incomplete'
    if phase['id']=='P05':state='RUNNING';reason='Receipt-derived reconstructed Main/Nested targets and independent P0/P1 contracts; strict source qualification remains separate'
    if phase['id']=='P06':state='RUNNING';reason='Train-only registered information/context preprocessing; receipt-derived track readiness, strict PIT remains separate'
    if phase['id'] in {'P07','P08','P09','P10'}:state='RUNNING';reason='Independent Auxiliary and Main/Nested development receipts below; completion does not imply strict final qualification'
    if phase['id'] in {'P12','P13'}:state='BLOCKED_DATA';reason='Scientific frozen final gates and P1 economics qualification absent; prediction diagnostics remain independent'
    if phase['id']=='P11':state='RUNNING';reason='Synthetic systems and actual process recovery pass; full development workload qualification incomplete'
    if phase['id']=='P14':state='RUNNING';reason='Incremental deterministic report executable; complete research report not qualified'
    if phase['id']=='P15':state='BLOCKED_DATA';reason='Operational freeze/candidate and actual future observation absent'
    from signalforge.track_status import continuation_phase_state
    state,reason=continuation_phase_state(repo,phase['id'],state,reason)
    phases[phase['id']]={'state':state,'reason':reason,'implementation_complete':False,
        'independent_branch_states':{'Auxiliary-C':aux.get('state','NOT_RUN'),**track_phase_states(repo,phase['id']),
            'economics':'BLOCKED_PRICE_P1','final':'BLOCKED_FREEZE_GATES'}}
atomic_json(folder/'phase_status.json',{'created_at':now(),'phases':phases})
control_report=folder/'auxiliary_invalid_cuda_controls.json';control_bill=folder/'auxiliary_invalid_cuda_controls_bill.json'
if control_report.exists() and control_bill.exists():
    control=json.loads(control_report.read_text());bill=json.loads(control_bill.read_text())
    if control.get('state')=='SUCCEEDED_NONPROMOTABLE_CUDA_CONTROLS' and len(control.get('checks',[]))==45 and bill.get('state')=='RESOURCE_ADMISSION_PENDING':
        # Reconcile the earlier successful version's mutable planning display;
        # immutable workflow/report snapshots retain its original pending bill.
        seconds=0.
        from signalforge.runtime import validate_bundle
        for check in control['checks']:
            directory=runtime/'artifacts/invalid_CUDA_controls'/check['artifact_id'];validate_bundle(directory)
            seconds+=json.loads((directory/'predictions.json').read_text())['seconds']
        bill.update(state='COMPLETED_VERIFIED_MODEL_BUNDLES',actual_control_bundles=45,
            sum_recorded_fit_predict_seconds=seconds,result_sha256=file_hash(control_report),
            charge_scope='Fit/predict sum is not the full resource-lease ledger wall charge; persistent compute ledger remains authoritative')
        atomic_json(control_bill,bill)
status=load('IMPLEMENTATION_STATUS.json')
status.update({'scope':'incremental_implementation_and_real_acquisition_checkpoint_NOT_COMPLETE','updated_at':now(),
               'implementation_complete':False,'real_data_acquisition_qualified':False,'strict_PIT_available':False,
               'single_gpu_qualified_on_user_device':True,'report_reproduced':True,
               'complete_research_report_reproduced':False,'reserved_evaluation_complete':False,
               'scientific_authorization_present':(repo/'.local/scientific_authorization.json').exists(),'freeze_receipt':None,
               'phase_status_path':'reports/phase_status.json','acceptance_mapping_path':'reports/acceptance_mapping.json',
               'tests_executed':sum(r['counts']['tests'] for r in tests['results']),
               'test_failures':sum((r['counts'].get('failures') or 0)+(r['counts'].get('errors') or 0) for r in tests['results']),
               'auxiliary_development_state':aux.get('state','NOT_RUN'),'auxiliary_successful_outer_fits':len(aux_success),
               'eia_development_completed':eia.get('completed',0),'eia_development_required':eia.get('required'),
               'acceptance_test_implemented':len(valid),'acceptance_test_passed':sum(c['test_result']=='PASSED' for c in cases),
               'resume_command':"pwsh -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\Resume-SourceChainV311.ps1",
               'remaining_implementation':['All source/acquisition readiness is established by current immutable receipts, not historical credential blockers',
                   'Main/Nested information readiness and CPU/GPU completion are receipt-derived; reconstructed development remains separate from strict final gates',
                   'N-PORT Tier-B accepted-time reconstruction is separate from original-publication evidence; P1 action/pay-date/open-clock audit remains independent',
                   'successor GPU admission is track-specific and uses a separate preregistered pilot ledger; parent v3 120-fit/43200-second ledgers remain unchanged',
                   'strict Tier-A source calibration/selection/freeze qualification remains required before the one authorized final batch',
                   'operational freeze and actual future observations; zero-shot and vectorized correctness tests do not constitute public research execution']})
status['test_source_scopes']={r['suite']:r.get('source_tree_hash') for r in tests['results']}
terminal_path=folder/'successor_continuation_terminal.json'
if terminal_path.exists():
    terminal=json.loads(terminal_path.read_text())
    status['resume_command']=None
    status['resume_gate']=terminal['reason']
    status['active_execution']={'state':terminal['state'],'path':'reports/successor_continuation_terminal.json'}
status['last_serial_full_validation_at']=last_full.get('created_at')
from signalforge.completion import current_full_validation
status['current_tree_full_validation_passed']=current_full_validation(repo)['passed']
status['active_validation']={'state':'SUCCEEDED' if status['current_tree_full_validation_passed'] and status['test_failures']==0 else 'STALE_OR_FAILED',
    'tests':status['tests_executed'],'receipt':'reports/test_execution.json','source_hash':code_hash(repo)}
freeze_path=repo/'.local/freeze_receipt.json'
if freeze_path.exists():
    try:
        from signalforge.integrity import verify_freeze
        receipt=json.loads(freeze_path.read_text(encoding='utf-8-sig'))
        auth=json.loads((repo/'.local/scientific_authorization.json').read_text(encoding='utf-8-sig'))
        freeze_id=verify_freeze(receipt,repo,runtime,auth)
        status.update(freeze_receipt=str(freeze_path),strict_PIT_available=True,freeze_state='VERIFIED_READY_FOR_FINAL')
        final_directory=runtime/'artifacts/final'/freeze_id
        if (final_directory/'receipt.json').exists():
            from signalforge.runtime import validate_bundle
            validate_bundle(final_directory)
            result=json.loads((final_directory/'evaluation.json').read_text())
            if result.get('status')!='FROZEN_BATCH_EXECUTED' or result.get('freeze_id')!=freeze_id:raise PermissionError('Final result identity/status mismatch')
            status.update(reserved_evaluation_complete=True,freeze_state='AUTHORIZED_FROZEN_BATCH_EXECUTED')
    except (PermissionError,RuntimeError,FileNotFoundError,ValueError,KeyError) as error:
        status['freeze_state']='BLOCKED_OR_INVALID';status['freeze_blocker']=str(error)
else:status['freeze_state']='BLOCKED_FREEZE_GATES'
status['scientific_admission_ledger_present']=(runtime/'ledger/final_access.json').exists()
for filename,key in [('auxiliary_execution_state.json','active_execution'),('source_integrity_incident.json','source_integrity_incident'),
                     ('eia_corrected_source_audit.json','corrected_source_audit'),('auxiliary_real_reload_qualification.json','cpu_model_reload'),
                     ('auxiliary_cpu_robustness.json','cpu_robustness'),
                     ('auxiliary_cpu_postprocessing.json','cpu_postprocessing'),
                     ('auxiliary_cpu_postprocessing_qualification.json','cpu_postprocessing_reload'),
                     ('remaining_workflow_state.json','remaining_workflow'),
                     ('canonical_materialization.json','canonical_cache_materialization'),
                     ('auxiliary_all_family_reload_qualification.json','all_family_reload'),
                     ('auxiliary_all_family_robustness.json','all_family_robustness'),
                     ('auxiliary_all_family_postprocessing.json','all_family_postprocessing'),
                     ('auxiliary_all_family_postprocessing_qualification.json','all_family_postprocessing_reload'),
                     ('auxiliary_capacity_results.json','capacity_development'),
                     ('auxiliary_capacity_qualification.json','capacity_reload'),
                     ('auxiliary_cpu_temporal_transport.json','cpu_temporal_transport'),
                     ('auxiliary_all_family_temporal_transport.json','all_family_temporal_transport'),
                     ('auxiliary_invalid_cpu_controls.json','invalid_cpu_controls'),
                     ('auxiliary_invalid_cuda_controls.json','invalid_cuda_controls'),
                     ('desktop_study_qualification.json','desktop_study_contracts'),
                     ('missingness_variant_audit.json','missingness_variant'),
                     ('research_figures.json','research_figures'),
                     ('track_input_qualification.json','track_input_qualification'),
                     ('experiment_execution_status.json','experiment_execution'),
                     ('frozen_gate_controls.json','frozen_gate_controls'),
                     ('real_systems_qualification.json','real_systems_diagnostic'),
                     ('real_training_systems_qualification.json','real_training_systems_diagnostic'),
                     ('real_process_recovery_qualification.json','real_process_recovery'),
                     ('fresh_cpu_reproduction.json','fresh_cpu_reproduction'),
                     ('eia_2023_calibration_source_audit.json','calibration_source_consistency'),
                     ('keyed_source_access.json','keyed_source_access'),
                     ('fred_segmented_plan.json','fred_segmented_plan'),
                     ('macro_canonical_integration.json','macro_canonical_integration'),
                     ('nport_bulk_integration.json','nport_bulk_integration'),
                     ('nport_bulk_integration_v311.json','nport_bulk_integration_v311'),
                     ('track_source_mapping_v31.json','track_source_mapping_v31'),
                     ('track_source_input_extension_v311.json','track_source_input_extension_v311'),
                     ('successor_gpu_bill_admission.json','successor_gpu_bill_admission'),
                     ('successor_gpu_full.json','successor_gpu_full'),
                     ('track_development_analysis.json','track_development_analysis'),
                     ('track_model_forensics.json','model_forensics'),
                     ('track_model_forensic_interpretation.json','model_forensic_interpretation'),
                     ('original_archive_evidence_research.json','original_archive_evidence_research'),
                     ('issuer_action_evidence_2023.json','partial_issuer_action_evidence'),
                     ('forward_diagnostic_admission_v32.json','forward_diagnostic_admission'),
                     ('calibration_Main_A.json','calibration_Main_A'),
                     ('calibration_Nested_B.json','calibration_Nested_B'),
                     ('strict_pit_final_gate_audit.json','strict_pit_final_gate_audit'),
                     ('successor_continuation_terminal.json','successor_continuation_terminal'),
                     ('track_input_qualification_v311.json','track_input_qualification_v311'),
                     ('successor_gpu_pilot_Main_A.json','successor_gpu_pilot_main'),
                     ('successor_gpu_pilot_Nested_B.json','successor_gpu_pilot_nested'),
                     ('fixed_gate_retrained.json','fixed_gate_retrained')]:
    path=folder/filename
    if path.exists():
        evidence=json.loads(path.read_text())
        status[key]={'state':evidence.get('state'),'path':'reports/'+filename}
incident_path=folder/'source_integrity_incident.json';reload_path=folder/'auxiliary_all_family_reload_qualification.json'
if incident_path.exists() and reload_path.exists() and aux.get('state')=='SUCCEEDED_DIAGNOSTIC' and aux.get('protocol',{}).get('compute_scope')=='all':
    reload=json.loads(reload_path.read_text())
    if reload.get('state')=='SUCCEEDED_ALL_FAMILY_RELOAD' and reload.get('n_verified_outer_bundles')==len(aux_success):
        incident=json.loads(incident_path.read_text());incident.update(state='CORRECTED_ALL_FAMILY_DEVELOPMENT_RECOMPUTED_RELOAD_VERIFIED',
            corrected_outer_bundles=len(aux_success),all_family_reload_path='reports/auxiliary_all_family_reload_qualification.json',
            historical_original_vintage_qualified=False,updated_at=now())
        atomic_json(incident_path,incident)
        status['source_integrity_incident']={'state':incident['state'],'path':'reports/source_integrity_incident.json'}
atomic_json(repo/'IMPLEMENTATION_STATUS.json',status)
atomic_json(folder/'model_cards.json',{'status':'correctness_paths_tested_research_unqualified',
    'statistical':['historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage_distributed_lag'],
    'cuda_tree':['lightgbm','xgboost'],'cuda_neural':['mlp','gru','gru_d_style','tft_style','transformer','rgmf_linear_joint','rgmf_gru_joint','rgmf_transformer_joint'],
    'no_confirmatory_predictive_claim':True,'training_cutoff':'2018-2022 chronological development folds; inner-only selection' if aux_success else 'synthetic correctness only; real development not fitted yet',
    'real_development_state':aux.get('state','NOT_RUN'),'successful_outer_fits':len(aux_success),
    'registered_development_families':aux.get('protocol',{}).get('families',[]),
    'auxiliary_successful_outer_fits':len(aux_success),
    'main_nested_cpu_qualification':'reports/track_input_qualification_v311.json',
    'successor_execution_plan':'reports/successor_execution_plan.json',
    'successor_gpu_completion':'reports/successor_gpu_full.json',
    'main_calibration':'reports/calibration_Main_A.json','nested_calibration':'reports/calibration_Nested_B.json',
    'development_analysis':'reports/track_development_analysis.json',
    'forensic_diagnosis':'reports/track_model_forensic_interpretation.json',
    'original_forecasts_retained':True,
    'strict_pit_gate_audit':'reports/strict_pit_final_gate_audit.json',
    'limitations':['compact style implementations','two-stage mature prequential adapter tested; all-track research integration incomplete',
                   'Auxiliary fixed development grid completed when its result state succeeds; all-track HPO and strict source qualification remain incomplete',
                   'Historical Auxiliary GRU-D path has inactive missingness decay; generic actual context-day correction has no public Main/Nested fit yet']})
atomic_json(folder/'data_cards.json',{'source_manifest':'reports/public_pilot.json','eia_manifest':'reports/eia_development_acquisition.json',
    'tiingo_manifest':'reports/tiingo_development_acquisition.json','tiingo_inputs':'reports/tiingo_input_build.json',
    'tiingo_claim_boundary':'Raw unadjusted P0 only, reconstructed 20:30 America/New_York EOD availability upper bound; same-Friday final EOD unavailable at Friday 18:00 origin; Tier B, original publication unqualified; no P1/final promotion',
    'nport_manifest':'reports/nport_development_acquisition.json',
    'source_extension':'reports/track_source_input_extension_v311.json',
    'macro_manifest':'reports/macro_canonical_integration.json',
    'nport_v311_manifest':'reports/nport_bulk_integration_v311.json',
    'cftc_manifest':'reports/cftc_canonical_integration.json','eia_market_manifest':'reports/eia_market_integration.json',
    'original_archive_evidence':'reports/original_archive_evidence_research.json',
    'partial_issuer_distribution_evidence':'reports/issuer_action_evidence_2023.json',
    'tier_A':False,'historical_PIT_qualified':False,'cftc_semantics':'positioning contracts, not reported cash flow',
    'eia_tier':'B reconstructed date-bound; original archived vintage unverified','main_targets_qualified':False,'p1_prices_actions_qualified':False})
future_cards=[
    {'observed_limitation':'Original archived values/publication timing unverified','new_hypothesis':'Observed incremental effects survive strict original vintage evidence',
     'needed_data':'Authenticated original release snapshots and known-at mappings','baseline':'age-aware all-information LightGBM CUDA',
     'experiment':'new preregistered A-only vs reconstructed comparison on an unused cohort','budget':'bounded acquisition pilot then measured bill within local 43200 GPU seconds',
     'success_failure_criterion':'paired date effect and calibration guardrails fixed before final','claim_boundary':'conditional forecasting, not causal flow alpha'},
    {'observed_limitation':'BF16/compile setup costs dominate tiny synthetic path','new_hypothesis':'Larger real development batches amortize cold costs without numerical-quality loss',
     'needed_data':'Qualified real development workload and same frozen inputs','baseline':'eager FP32 CUDA',
     'experiment':'counterbalanced repeated end-to-end timing including cold setup, recovery and losses','budget':'admit from measured pilot inside 43200 seconds',
     'success_failure_criterion':'faster time-to-valid-result with fixed numerical tolerance','claim_boundary':'local one-GPU workload; no distributed or investment-performance inference'},
    {'observed_limitation':'The CPU permutation control did not establish predictive skill: paired confidence interval included zero',
     'new_hypothesis':'Release-aware gains are distinguishable from seasonal persistence with a larger independent release sample',
     'needed_data':'New unused release dates with authenticated original vintages, not repeated existing contexts',
     'baseline':'historical seasonal persistence and same-information age-aware CUDA trees',
     'experiment':'preregister paired date-block effects and null controls in a new study',
     'budget':'new bounded acquisition and measured compute bill; no automatic current budget increase',
     'success_failure_criterion':'fixed positive paired effect with uncertainty and null controls; inconclusive intervals remain inconclusive',
     'claim_boundary':'forecasting evidence only, no causal or economic claim'},
    {'observed_limitation':'Core GRU-D-style path has no missing raw contexts and its elapsed tensors are zero; decay value is not identified',
     'new_hypothesis':'Actual clock-derived missingness decay improves outage calibration beyond the same-information GRU',
     'needed_data':'New unused authentic release contexts with genuine missingness and ordered clock lineage',
     'baseline':'same-information GRU and age-aware LightGBM CUDA',
     'experiment':'new registered actual-context-day GRU-D comparison, distinct retrained and frozen outages, no source/date selection',
     'budget':'separate measured pilot and admitted bill; no automatic current pilot or compute ceiling increase',
     'success_failure_criterion':'fixed paired date-block improvement plus calibration/coverage guards; inactive decay remains uninformative',
     'claim_boundary':'conditional prediction only; compact input decay is not full GRU-D reproduction'}]
for card_path in sorted((repo/'research/future_research').glob('*.json')):
    future_cards.append(json.loads(card_path.read_text()))
for card in future_cards:
    card.update(new_protocol='separate registered successor version; never overwrite sgqx-v3',
                cohort_policy='new_unused_or_post_actual_freeze_prospective',reuse_consumed_final=False)
from signalforge.research_cards import validate_future_cards
validate_future_cards(future_cards)
atomic_json(folder/'future_research_cards.json',future_cards)
print(json.dumps({'tests_executed':status['tests_executed'],'mapped_requirements':len(valid),'processed_audit':audit},indent=2))
