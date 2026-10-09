"""Actual completed development evidence, distinct from synthetic acceptance."""
import json
from pathlib import Path
import numpy as np
from signalforge.runtime import paths,validate_bundle,digest,file_hash

def load(name):
    repo,_=paths();return json.loads((repo/'reports'/name).read_text(encoding='utf-8-sig'))

def test_real_all_information_and_capacity_baselines_share_grid_and_metadata():
    repo,runtime=paths();core=load('auxiliary_development_results.json');cap=load('auxiliary_capacity_results.json')
    reference={}
    for r in core['results']:
        if r['family']!='lightgbm':continue
        directory=runtime/'artifacts/auxiliary'/r['run_id'];validate_bundle(directory)
        transform=json.loads((directory/'transform.json').read_text())
        assert r['feature_variant'] is None and len(transform['center'])==14
        p=json.loads((directory/'predictions.json').read_text())
        reference[r['outer_year'],r['seed']]=(p,r['scale'],r['normalizer_id'],transform)
    assert len(cap['results'])==30 and cap['protocol']['data_id']==core['results'][0]['data_id']
    for r in cap['results']:
        directory=runtime/'artifacts/auxiliary'/r['run_id'];validate_bundle(directory)
        p=json.loads((directory/'predictions.json').read_text());old,scale,normalizer,transform=reference[r['outer_year'],r['seed']]
        import pandas as pd
        assert list(pd.to_datetime(p['decision_time'],utc=True))==list(pd.to_datetime(old['decision_time'],utc=True))
        assert p['y']==old['y'] and r['scale']==scale and r['normalizer_id']==normalizer
        assert json.loads((directory/'transform.json').read_text())==transform
        assert r['capacity_spec']['relative_parameter_gap']<=r['capacity_spec']['tolerance']==.01
        assert r['actual_CUDA_used'] and len(r['inner_attempts'])==20

def test_real_nontrivial_baseline_search_has_complete_exact_recipe_proof():
    core=load('auxiliary_development_results.json');proof=load('canonical_cache_upgrade_qualification.json')
    assert core['protocol']['full_trials']==20 and core['protocol']['epochs']==100
    assert proof['all_registered_recipes_required']==proof['all_registered_recipes_reusable']==1952
    assert not proof['missing_recipe_ids'] and proof['new_model_fits']==0
    from signalforge.development import trial_grid
    for family in ['ridge','linear_quantile','lightgbm','xgboost','mixed_frequency_shrinkage']:
        grid=trial_grid(family)
        assert len(grid)==20 and len({t['regularization'] for t in grid})==20
        assert min(t['regularization'] for t in grid)==.0001 and max(t['regularization'] for t in grid)==10000

def test_real_null_targets_and_invalid_controls_are_retained_nonpromotable():
    _,runtime=paths();control=load('auxiliary_invalid_cuda_controls.json')
    assert len(control['checks'])==45 and not control['protocol']['promotion_eligible'] and not control['reserved_access']
    for c in control['checks']:validate_bundle(runtime/'artifacts/invalid_CUDA_controls'/c['artifact_id'])
    table={r['control']:r for r in control['comparison_table']}
    null=table['train_target_permutation']
    assert null['n_market_dates']==260 and null['control_pinball']>0
    # Diagnostic interpretation, never an outcome-selected scientific threshold.
    assert null['paired_reference_minus_control']['lower_95']<=0<=null['paired_reference_minus_control']['upper_95']
    assert set(table)=={'train_target_permutation','wrong_release_clock','future_target_sentinel'}

def test_real_compile_includes_cold_cost_and_requires_numeric_parity():
    doc=load('real_training_systems_qualification.json');probe=doc['probes']['compile']
    assert probe['passed'] and not probe['adopted'] and doc['actual_CUDA_used']
    assert probe['cold_seconds']>0 and probe['seconds_including_cold']>=probe['cold_seconds']
    assert probe['max_abs_delta']<=doc['protocol']['FP32_equivalence_tolerance']
    inference=load('real_systems_qualification.json')['probes']['compile']
    assert inference['passed'] and inference['cold_cost_included'] and not inference['adopted']

def test_real_microbatches_preserve_full_effective_optimizer_batch():
    doc=load('real_training_systems_qualification.json')
    for name in ['microbatch32','microbatch128']:
        p=doc['probes'][name]
        assert p['passed'] and not p['adopted']
        assert p['max_abs_delta']<=doc['protocol']['FP32_equivalence_tolerance']
        assert p['distinct_training_dates']==doc['protocol']['n_mature_training_dates']
        assert p['useful_examples']==p['distinct_training_dates']*doc['protocol']['epochs']

def test_real_useful_counts_and_spawn_workers_do_not_inflate_sample_size():
    doc=load('real_training_systems_qualification.json');n=doc['protocol']['n_mature_training_dates']
    probes=doc['probes']['dataloader_workers_and_pinned_copy']
    assert [p['workers'] for p in probes]==[0,2,4]
    assert all(p['passed'] and p['useful_examples']==n for p in probes)
    repo,_=paths()
    import ast
    tree=ast.parse((repo/'scripts/benchmark_training_systems.py').read_text())
    contexts=[k.value for node in ast.walk(tree) if isinstance(node,ast.Call) for k in node.keywords if k.arg=='multiprocessing_context']
    assert len(contexts)==1 and isinstance(contexts[0],ast.IfExp) and contexts[0].body.value=='spawn'

def test_real_source_bug_preserves_originals_and_corrects_every_core_model():
    _,runtime=paths();incident=load('source_integrity_incident.json')
    assert incident['state']=='CORRECTED_ALL_FAMILY_DEVELOPMENT_RECOMPUTED_RELOAD_VERIFIED'
    assert incident['corrected_outer_bundles']==270 and incident['already_known_target_count_after']==0
    archive=runtime/'artifacts/incidents/eia_2019_duplicate_original_inputs';validate_bundle(archive)
    reload=load('auxiliary_all_family_reload_qualification.json')
    assert reload['n_verified_outer_bundles']==270 and not reload['failures']
    assert all(c['max_mean_abs_delta']<=c['tolerance'] and c['max_quantile_abs_delta']<=c['tolerance'] for c in reload['checks'])

def test_real_fresh_environment_reproduces_declared_cpu_evidence_tier():
    _,runtime=paths();doc=load('fresh_cpu_reproduction.json')
    assert doc['state']=='PASSED_FRESH_PACKAGE_ENVIRONMENT_TIER_B_CPU_REPRODUCTION'
    directory=runtime/'artifacts/fresh_cpu_reproduction'/doc['protocol_id'];validate_bundle(directory)
    assert doc['fresh_site_packages'] and doc['new_model_fits']==doc['CUDA_allocations']==0
    assert len(doc['checks'])==75 and all(c['max_prediction_abs_delta']<=1e-10 and c['max_loss_abs_delta']<=1e-12 for c in doc['checks'])
    assert not doc['complete_research_reproduction'] and not doc['reserved_access']

def test_real_rootless_source_build_and_physical_host_disk_lineage():
    repo,runtime=paths();doc=load('rootless_install_evidence.json')
    validate_bundle(runtime/'artifacts/rootless_install_audits'/doc['artifact_id'])
    assert doc['global_driver_or_configuration_mutations']==0 and doc['source_archive_original_hash_retained']
    assert doc['retained_source_build_wheels'] and all(w['installed_library_exact_match'] for w in doc['retained_source_build_wheels'])
    disk=load('host_disk_evidence.json');config=json.loads((repo/'.local/runtime.json').read_text(encoding='utf-8-sig'))
    assert disk['distro']==config['distro'] and disk['volume']==config['wsl_host_volume'].rstrip(':\\')
    assert disk['wsl_base_path'].startswith(disk['volume']+':\\') and 'verified WSL VHDX' in disk['source']
    assert disk['free_bytes']>0

def test_real_all_registered_results_and_negative_controls_are_retained():
    core=load('auxiliary_development_results.json');analysis=load('auxiliary_development_analysis.json')
    assert set(r['family'] for r in analysis['comparison_table'])==set(core['protocol']['families'])
    assert len(core['results'])==270 and all(r['n_unique_dates']>0 for r in core['results'])
    robustness=load('auxiliary_all_family_robustness.json')
    assert len(robustness['comparison_table'])==40 and robustness['n_market_dates']==260
    assert robustness['no_refitting_or_model_seed_selection'] and not robustness['final_access']
    fixed=load('fixed_gate_retrained.json')
    assert len(fixed['results'])==45 and len(fixed['comparison_table'])==3 and not fixed['qualified_for_final']

def test_real_owned_process_interrupt_restores_actual_optimizer_and_rng():
    _,runtime=paths();doc=load('real_process_recovery_qualification.json')
    assert doc['state']=='PASSED_ACTUAL_DEVELOPMENT_OWN_PROCESS_RECOVERY'
    directory=runtime/'artifacts/real_process_recovery'/digest(doc['protocol']);validate_bundle(directory)
    validate_bundle(directory/'interruption_boundary')
    signal=json.loads((directory/'interruption_boundary/signal.json').read_text())
    assert signal['signal']=='SIGTERM' and signal['exit_code']==-15
    assert doc['interrupted_step']==50 and doc['final_steps']==100 and doc['sample_cursor']==50*doc['n_distinct_training_dates']
    assert doc['loss_traces_exactly_equal'] and doc['max_prediction_abs_delta']==doc['max_parameter_abs_delta']==0

def test_real_posthoc_cases_remain_exploratory():
    robustness=load('auxiliary_all_family_robustness.json')
    assert robustness['no_exact_causal_PnL_attribution'] and len(robustness['error_cases'])==5
    assert all('post-hoc' in case['selection'] and 'never changes the cohort/models' in case['selection'] for case in robustness['error_cases'])
    assert 'Exploratory' in robustness['protocol']['claim_boundary']
    analysis=load('auxiliary_development_analysis.json')
    assert analysis['exploratory_multiplicity']['family_size']==17
    assert 'no H1/H2/H3, final, economics or causal claim' in analysis['claim_boundary']

def test_real_same_release_pdf_cannot_hide_calibration_source_inconsistency():
    repo,runtime=paths();audit=load('eia_2023_calibration_source_audit.json')
    validate_bundle(runtime/'artifacts/eia_calibration_failure_audits'/audit['artifact_id'])
    assert audit['state']=='BLOCKED_SOURCE_CONSISTENCY' and not audit['substitution_admitted'] and not audit['core_inputs_changed']
    assert set(audit['registered_parser_failures'])=={'csv','pdf'}
    checks=[c for c in audit['checks'] if c['field']=='Commercial (Excluding SPR)']
    assert len(checks)==2 and all(abs(c['level_difference_minus_reported_change'])>c['tolerance'] for c in checks)
    assert file_hash(repo/audit['visual_verification']['image'])==audit['visual_verification']['sha256']
