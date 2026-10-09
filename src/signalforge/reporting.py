"""Deterministic incremental reporting from checksum-validated small evidence files."""
import html
import json
import os
import hashlib
from pathlib import Path
from .runtime import atomic_json,atomic_bytes,file_hash,code_hash,commit_bundle,digest,now


def render(repo):
    repo=Path(repo)
    runtime=Path(os.environ['SIGNALFORGE_RUNTIME']) if (os.environ.get('SIGNALFORGE_RUNTIME') and
        Path(os.environ.get('SIGNALFORGE_REPO','')).resolve()==repo.resolve()) else repo/'reports'
    import fcntl
    lock=runtime/'locks/report.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _render(repo,runtime)


def _render(repo,runtime):
    repo=Path(repo);folder=repo/'reports';folder.mkdir(exist_ok=True)
    names=['public_pilot.json','test_execution.json','systems_qualification.json','environment_audit.json',
           'eia_development_acquisition.json','processed_data_audit.json','acceptance_mapping.json','phase_status.json',
           'raw_data_audit.json','eia_unit_qualification.json','process_recovery_qualification.json','source_document_verification.json',
           'host_disk_evidence.json','cftc_development_acquisition.json','cftc_acquisition_plan.json','continuation_state.json',
           'auxiliary_registered_protocol.json','auxiliary_pilot_costs.json','auxiliary_measured_run_bill.json',
           'auxiliary_development_results.json','auxiliary_development_analysis.json',
           'source_integrity_incident.json','eia_corrected_source_audit.json','source_correction_cache_audit.json',
           'auxiliary_cpu_robustness.json','auxiliary_error_cases.json']
    names+=['eia_raw_eda.json','gpu_runtime_context.json','run_plan.json','auxiliary_execution_state.json']
    names+=['cpu_test_execution.json','canonical_cache_upgrade_qualification.json','training_import_boundary.json']
    names+=['auxiliary_cpu_postprocessing.json','auxiliary_cpu_postprocessing_qualification.json']
    names+=['remaining_workflow_state.json','canonical_materialization.json','auxiliary_all_family_reload_qualification.json',
            'auxiliary_all_family_robustness.json','auxiliary_all_family_postprocessing.json',
            'auxiliary_all_family_postprocessing_qualification.json','auxiliary_all_family_postprocessing_bill.json']
    names+=['auxiliary_invalid_cpu_controls.json']
    names+=['auxiliary_invalid_cuda_controls_protocol.json','auxiliary_invalid_cuda_controls_bill.json','auxiliary_invalid_cuda_controls.json']
    names+=['research_figures.json','auxiliary_capacity_results.json','auxiliary_capacity_qualification.json','auxiliary_capacity_bill.json']
    names+=['auxiliary_cpu_temporal_transport.json','auxiliary_all_family_temporal_transport.json']
    names+=['track_input_qualification.json','experiment_execution_status.json']
    names+=['real_systems_protocol.json','real_systems_qualification.json']
    names+=['real_training_systems_qualification.json','eia_publication_exceptions.json']
    names+=['fixed_gate_retrained_protocol.json','fixed_gate_retrained_bill.json','fixed_gate_retrained.json']
    names+=['frozen_gate_controls_protocol.json','frozen_gate_controls.json']
    names+=['final_evaluation.json']
    names+=['runtime_inventory_qualification.json']
    names+=['model_cards.json','data_cards.json','future_research_cards.json']
    names+=['alfred_public_export_access.json']
    names+=['contamination_register.json']
    names+=['desktop_study_qualification.json']
    names+=['missingness_variant_audit.json']
    names+=['rootless_install_evidence.json']
    names+=['native_archive_test_execution.json']
    names+=['real_process_recovery_plan.json','real_process_recovery_qualification.json']
    names+=['fresh_cpu_reproduction_plan.json','fresh_cpu_reproduction.json']
    names+=['eia_2023_calibration_source_audit.json']
    names+=['keyed_source_access.json','fred_development_acquisition.json','fred_segmented_plan.json','macro_canonical_integration.json']
    names+=['tiingo_development_acquisition.json','tiingo_action_acquisition.json','tiingo_input_build.json','nport_development_acquisition.json','nport_bulk_integration.json','nport_bulk_integration_v311.json','data_unblock_checkpoint.json']
    names+=['completion_extension_v31.json','track_calibration_status.json','track_source_mapping_v31.json','track_source_input_extension_v311.json','track_input_qualification_v311.json']
    names+=['successor_gpu_pilot_Main_A.json','successor_gpu_pilot_Nested_B.json','successor_gpu_pilot.json']
    names+=['successor_execution_plan.json','successor_gpu_bill_admission.json','successor_gpu_full_Main_A.json','successor_gpu_full_Nested_B.json','successor_gpu_full.json',
            'track_development_analysis.json','calibration_Main_A.json','calibration_Nested_B.json','strict_pit_final_gate_audit.json','successor_continuation_terminal.json']
    names+=['fred_cached_page_audit.json','takeover_development_checkpoint.json','development_budget_checkpoint.json','track_cpu_gpu_budget_accounting_incident_v311.json']
    names+=['track_model_forensics.json','track_model_forensic_interpretation.json','original_archive_evidence_research.json','issuer_action_evidence_2023.json','forward_diagnostic_admission_v32.json','forensic_continuation_checkpoint.json','forensic_figures.json']
    captured={name:(folder/name).read_bytes() for name in names if (folder/name).exists()}
    if (repo/'IMPLEMENTATION_STATUS.json').exists():captured['IMPLEMENTATION_STATUS.json']=(repo/'IMPLEMENTATION_STATUS.json').read_bytes()
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in captured.items()}
    implementation=code_hash(repo);snapshot_id=digest({'code_hash':implementation,'inputs':hashes})
    snapshot=runtime/'artifacts/report_snapshots'/snapshot_id
    live_sources={name:str(repo/name if name=='IMPLEMENTATION_STATUS.json' else folder/name) for name in captured}
    receipt=commit_bundle(snapshot,captured,{'code_hash':implementation,'live_sources':live_sources})
    evidence={name:{'path':str(snapshot/name),'sha256':hashes[name],'live_source_path':live_sources[name]} for name in captured}
    documents={name:json.loads(content) for name,content in captured.items()}
    if 'future_research_cards.json' in documents:
        from .research_cards import validate_future_cards
        validate_future_cards(documents['future_research_cards.json'])
    figures=documents.get('research_figures.json',{})
    if figures.get('repo_snapshot_directory'):
        for name,checksum in figures['figures'].items():
            path=(repo/figures['repo_snapshot_directory']/name).resolve()
            if not path.is_relative_to((repo/'reports/figures/snapshots').resolve()) or file_hash(path)!=checksum:
                raise ValueError('Scientific figure snapshot missing or changed')
            evidence['figure:'+name]={'path':str(path),'sha256':checksum,'live_source_path':str(path)}
    final=documents.get('final_evaluation.json',{})
    if final.get('status')=='FROZEN_BATCH_EXECUTED':
        from .runtime import validate_bundle
        final_directory=runtime/'artifacts/final'/final['freeze_id'];validate_bundle(final_directory)
        if json.loads((final_directory/'evaluation.json').read_text())!=final:raise ValueError('Final report differs from immutable authorized evaluation')
        final_boundary='The authorized frozen candidate batch executed; its immutable final evaluation is retained. No model/seed/threshold was selected from these outcomes. [final_evaluation.json]'
    elif (runtime/'ledger/final_access.json').exists():
        final_boundary='Scientific batch admission is recorded, but a complete immutable final evaluation is absent. Admission alone does not assert successful evaluation.'
    else:final_boundary='Reserved evaluation remains sealed; genuine all-track READY_FOR_FINAL gates remain incomplete.'
    lines=['# SignalForge-QX incremental technical report','',
           'Status: INCOMPLETE. This report does not assert completion of the master workflow.',
           final_boundary,
           'P1 economics is BLOCKED_PRICE_P1. No qualified daily raw price/action panel is available.',
           '', '## Software and evidence boundary','',
           'Implemented: durable JSON/bundle validation, bounded official-source acquisition, CFTC/EIA/FRED parsing primitives, ',
           'PIT selection, maturity/purge/train transforms, separate mean/quantile statistical and CUDA tree adapters, compact neural/RGMF joint architectures, ',
           'fixed-grid fallback scoring, paired block CI, Holm, support-gated calibration, calendar/P1 action oracles, share/cash ledger and constrained controller.',
           'N-PORT offline reference-month/dissemination/amendment/fixed-cohort parser and mature prequential residual CUDA adapters are tested. Live SEC timing reconstruction remains unqualified.',
           'Unfinished: complete all-track integration, all E01–E10 research execution, ',
           'all acceptance requirements and all-track qualification/integration. Generic gated freeze/final and forward orchestration exists but remains research-unqualified. Compact paths are not full paper reproductions.',
           '', '## Data audit','']
    if 'public_pilot.json' in documents:
        for source in documents['public_pilot.json']['sources']:
            if 'raw' in source:
                extra=''
                if 'audit' in source:
                    a=source['audit'];extra=f"; rows {a['rows']}, entities {a['entities']}, report dates {a['dates']}"
                lines.append(f"- {source['source']} {source.get('family',source.get('table',''))}: {source['raw']['bytes']} bytes, SHA256 `{source['raw']['sha256']}`{extra}. Raw acquisition; historical PIT/model qualification not complete. [public_pilot.json]")
            else:
                lines.append(f"- {source['source']}: {source['state']}; {source['reason']}. [public_pilot.json]")
    if 'eia_development_acquisition.json' in documents:
        d=documents['eia_development_acquisition.json']
        lines.extend(['',f"EIA development archive checkpoint: {d['completed']}/{d['required']} officially discovered issues parsed; state {d['state']}. [eia_development_acquisition.json]",
                      'The available CSV archive panel begins at the first discovered dated path; earlier coverage is not assumed. Tier B keeps the conservative date bound and original-vintage uncertainty visible.'])
    if 'IMPLEMENTATION_STATUS.json' in documents:
        d=documents['IMPLEMENTATION_STATUS.json']
        lines.extend(['','### Separate completion dimensions','',
            '| Dimension | Verified checkpoint state |','|---|---|',
            f"| Complete implementation | {d.get('implementation_complete',False)} |",
            f"| Qualified real public-data acquisition | {d.get('real_data_acquisition_qualified',False)} |",
            f"| Strict PIT available | {d.get('strict_PIT_available',False)} |",
            f"| Actual local CUDA capability | {d.get('single_gpu_qualified_on_user_device',False)} |",
            f"| Reserved batch completed | {d.get('reserved_evaluation_complete',False)} |",
            f"| Final freeze | {d.get('freeze_state','BLOCKED_FREEZE_GATES')} |",
            '| P1 economic results | BLOCKED_PRICE_P1 |',
            '| Real prospective outcomes | NOT_OBSERVED |',
            '| AWS/physical multi-GPU operation | NOT_EXECUTED |',
            'Capability, artifact completion and scientific qualification are different dimensions. [IMPLEMENTATION_STATUS.json]'])
    if 'cftc_development_acquisition.json' in documents:
        c=documents['cftc_development_acquisition.json']
        lines.extend(['',f"CFTC annual development archives: {c['completed']}/{c['required']} raw archives qualified; historical original-release timing/vintages remain unqualified. [cftc_development_acquisition.json]",
                      'No nominal Friday timestamp is substituted for missing actual-release evidence. Positions remain contract counts and positioning measures, never dollar fund cash flow.'])
    if 'alfred_public_export_access.json' in documents:
        d=documents['alfred_public_export_access.json']
        lines.extend(['',f"Official keyless ALFRED vintage export check: {d['state']}; configured WSL and bounded native documentation requests failed. No observation bytes or model inputs were committed. The keyed API remains independently blocked by the missing supplied key. [alfred_public_export_access.json]"])
    if 'eia_unit_qualification.json' in documents:
        if 'eia_publication_exceptions.json' in documents:
            d=documents['eia_publication_exceptions.json']
            lines.extend(['',f"EIA June 2022 publication exception: {d['state']}; June 17 and June 24 reference-week data were published together on June 29. The supplemental table is retained separately; no backdating to June 23 or change to the registered main-issue target. [eia_publication_exceptions.json]",d['archive_count_scope']+'.'])
        if 'real_training_systems_qualification.json' in documents:
            d=documents['real_training_systems_qualification.json']
            lines.extend(['',f"Real CUDA training systems diagnostic: {d['state']}; one fixed MLP workload, {d['protocol']['n_mature_training_dates']} mature training dates, 100 epochs per profiling replica. Precision, compilation, accumulation, optimizer-state recovery, workers, preprocessing cache and CPU overlap results are retained. No optimization or replica was selected for scientific promotion. [real_training_systems_qualification.json]"])
        unit=documents['eia_unit_qualification.json']
        lines.extend(['',f"EIA table 4 PDF unit evidence: million barrels present={unit['extraction']['million_barrels_present']}, visually verified={unit['visual_verified']}; source SHA256 `{unit['source']['sha256']}`. [eia_unit_qualification.json]",
                      'The 2022-01-12 official footnote identifies the commercial-crude lease-stock exclusion beginning with the week ended 2016-10-07. This measurement break is retained for audit, not used as a retroactively available model feature or a reason to delete stressed rows. Printed values are rounded; CSV precision and unrounded differences are preserved. [eia_unit_qualification.json]'])
    lines.extend(['','## Test evidence',''])
    for result in documents.get('test_execution.json',{}).get('results',[]):
        lines.append(f"- {result['suite']}: {result['counts']}; exit {result['exit_code']}; {result['seconds']:.3f} seconds. Synthetic fixture/correctness evidence. [test_execution.json]")
    if 'cpu_test_execution.json' in documents:
        lines.append('Newer CPU-only validation is separate from the last serial CUDA suite; source hashes identify each implementation checkpoint. Full CUDA checks wait for the heavy lease to end.')
        for result in documents['cpu_test_execution.json']['results']:
            lines.append(f"- CPU checkpoint {result['suite']}: {result['counts']}; exit {result['exit_code']}; source `{result['source_tree_hash']}`. [cpu_test_execution.json]")
    if 'auxiliary_execution_state.json' in documents:
        execution=documents['auxiliary_execution_state.json'];progress=documents.get('auxiliary_development_results.json',{})
        count=sum(row.get('state')=='SUCCEEDED' for row in progress.get('results',[]))
        lines.extend(['',f"Captured development execution: {execution['state']}; {count} successful outer bundles in this snapshot. A running grid is incomplete. [auxiliary_execution_state.json, auxiliary_development_results.json]"])
    lines.extend(['','The 160 catalogue requirements remain separate. See acceptance_mapping.json for actual test linkage and unimplemented requirements.',
                  '', '## Model comparisons, ablations, calibration and economics','',
                  'NOT EXECUTED on a qualified full research panel. No predictive superiority, calibrated tail coverage, alpha, economic qualification or causal source P&L attribution is asserted.',
                  'Calibration support and ledger oracles are tests, not realized calibration/economic results.',
                  '', '## Single-GPU systems evidence',''])
    incident_path=folder/'source_integrity_incident.json'
    if 'source_integrity_incident.json' in documents:
        incident=documents['source_integrity_incident.json']
        lines.extend(['','Source integrity incident: '+incident['state']+'. The stale 2019-07-03 CSV was replaced by the same-release official PDF at published 0.1-million-barrel precision. Earlier fitted scores are invalid until corrected-data recomputation; immutable originals are retained. No reserved unblinding occurred. [source_integrity_incident.json]'])
    if 'auxiliary_development_analysis.json' in documents and documents['auxiliary_development_analysis.json'].get('source_integrity_valid',True):
        analysis=documents['auxiliary_development_analysis.json']
        lines.extend(['','### Reconstructed physical-only Auxiliary development diagnostic','',
                      analysis['claim_boundary'],'',
                      '| Family | State | Dates | Normalized pinball | Native seed/date coverage |',
                      '|---|---|---:|---:|---:|'])
        for row in analysis['comparison_table']:
            lines.append(f"| {row['family']} | {row['state']} | {row.get('n_unique_dates','')} | {row.get('pinball','')} | {row.get('native_seed_date_coverage','')} |")
        lines.extend(['','All registered seeds remain in the ensemble. Missing model/seed forecasts use the training historical fallback; no difficult decision dates are dropped.',
                      'See auxiliary_development_analysis.json for 4/8/13-week paired-block diagnostics, calibration coverage and unscored folds. Main/Nested confirmatory contrasts remain missing, not successful.'])
        if 'exploratory_multiplicity' in analysis:
            adjustment=analysis['exploratory_multiplicity']
            lines.extend(['',f"Exploratory architecture multiplicity: {adjustment['family_size']} full registered contrasts; {analysis.get('searched_inner_recipes')} earlier-inner recipes searched. BH requires independence/PRDS; BY is retained as a conservative dependence sensitivity. Both depend on the block p-value assumptions and establish no confirmatory result.",
                '| Exploratory contrast | BH | BY sensitivity |','|---|---:|---:|'])
            for name in adjustment['registered_family']:lines.append(f"| {name} | {adjustment['BH'][name]} | {adjustment['BY_arbitrary_dependence'][name]} |")
            lines.append('Nonsignificance is not equivalence; distinct dates and MDE limits are preserved. [auxiliary_development_analysis.json]')
    if 'auxiliary_cpu_robustness.json' in documents:
        if 'missingness_variant_audit.json' in documents:
            missing=documents['missingness_variant_audit.json']
            lines.extend(['','### Missingness variant limitation','',missing['interpretation'],
                'The audited core contexts contain '+str(missing['raw_context_missing_values'])+' missing raw values. Actual stored forecast deltas and receipt identities are retained; inactive decay cannot establish missingness modeling value. [missingness_variant_audit.json]'])
        robustness=documents['auxiliary_cpu_robustness.json']
        lines.extend(['','### Fixed CPU outage diagnostics','',
                      'Exploratory frozen-input interventions; no refitting, seed selection or causal P&L attribution.',
                      '| Family | Scenario | Original pinball | Perturbed pinball |',
                      '|---|---|---:|---:|'])
        for row in robustness['comparison_table']:
            lines.append(f"| {row['family']} | {row['scenario']} | {row['original_pinball']:.6f} | {row['perturbed_pinball']:.6f} |")
        lines.append('Post-hoc largest-loss cases retain source/target hashes, model receipts, publication clocks and the blocked controller state in auxiliary_cpu_robustness.json.')
    if 'auxiliary_cpu_postprocessing.json' in documents:
        post=documents['auxiliary_cpu_postprocessing.json']
        lines.extend(['','### CPU 2023 selection/calibration diagnostic','',
                      'Training ends in 2022; hyperparameters come only from the earlier 2022 inner fold. All five registered seeds have equal weights. No scientific winner is chosen while CUDA comparisons are unfinished.',
                      '| Family | Window | Forecast dates | Mature qualified targets | Raw normalized pinball |',
                      '|---|---|---:|---:|---:|'])
        for row in post['comparison_table']:
            lines.append(f"| {row['family']} | {row['window']} | {row['forecast_dates']} | {row['mature_qualified_target_dates']} | {row['raw_normalized_pinball']:.6f} |")
        lines.append('Missing/immature targets remain explicit. Quantile corrections require their registered minimum support; unsupported quantiles use identity. Calibration-window metrics are fitted-window diagnostics. These artifacts are retrospective Tier B and unqualified for final/forward/economics. [auxiliary_cpu_postprocessing.json]')
    if 'remaining_workflow_state.json' in documents:
        workflow=documents['remaining_workflow_state.json']
        lines.extend(['','### Remaining authorized workflow','',f"State: {workflow['state']}; active stage: {workflow.get('active_stage','waiting')}. [remaining_workflow_state.json]"])
        for stage in workflow.get('stages',[]):lines.append(f"- {stage['stage']}: {stage['state']}; exit {stage.get('exit_code','not applicable')}; log {stage.get('log','not applicable')}")
        if workflow.get('final_gate'):lines.append('Scientific final gate: '+str(workflow['final_gate']))
    if 'auxiliary_invalid_cpu_controls.json' in documents:
        controls=documents['auxiliary_invalid_cpu_controls.json']
        lines.extend(['','### Nonpromotable real CPU controls','',
            'Fixed earlier-inner Ridge recipes; all three seeds retained. Wrong publication clocks and future-target sentinels are deliberately invalid Tier C controls, never candidate models.',
            '| Control | Reference pinball | Control pinball |','|---|---:|---:|'])
        for row in controls['comparison_table']:lines.append(f"| {row['control']} | {row['reference_pinball']:.6f} | {row['control_pinball']:.6f} |")
        lines.append('The null permutation comparison and its paired interval are retained even when the null has lower mean loss. An apparent fit gain is insufficient evidence of useful conditional prediction. Authenticated latest-revision controls remain blocked. [auxiliary_invalid_cpu_controls.json]')
    if 'auxiliary_invalid_cuda_controls.json' in documents:
        controls=documents['auxiliary_invalid_cuda_controls.json']
        lines.extend(['','### Nonpromotable actual CUDA tree controls','',
            'Matching earlier-inner LightGBM recipes, all three seeds and the full 260-date grid. Training used CUDA; saved tree prediction used the host backend. These controls cannot enter selection or final evaluation.',
            '| Control | Reference pinball | Control pinball | Paired 95% interval |','|---|---:|---:|---|'])
        for row in controls['comparison_table']:
            paired=row['paired_reference_minus_control']
            lines.append(f"| {row['control']} | {row['reference_pinball']:.6f} | {row['control_pinball']:.6f} | [{paired['lower_95']:.6f}, {paired['upper_95']:.6f}] |")
        lines.append('Permutation is a null diagnostic. Wrong clocks and future-target sentinels are deliberately invalid leakage probes. Latest-revision controls remain blocked by missing authenticated vintages. [auxiliary_invalid_cuda_controls.json]')
    if 'auxiliary_all_family_postprocessing.json' in documents:
        post=documents['auxiliary_all_family_postprocessing.json']
        lines.extend(['','### All-family 2023 diagnostic','',
            'Fixed 2022 inner-only hyperparameters and five equal-weight seeds. Missing source targets remain in forecast denominators. Tier B results cannot authorize the strict all-track final.',
            '| Family | Window | Forecast dates | Mature targets | Raw normalized pinball |','|---|---|---:|---:|---:|'])
        for row in post['comparison_table']:lines.append(f"| {row['family']} | {row['window']} | {row['forecast_dates']} | {row['mature_qualified_target_dates']} | {row['raw_normalized_pinball']:.6f} |")
    if 'auxiliary_all_family_robustness.json' in documents:
        robustness=documents['auxiliary_all_family_robustness.json']
        lines.extend(['','### All-family fixed outage diagnostic','',
            '| Family | Scenario | Original pinball | Perturbed pinball |','|---|---|---:|---:|'])
        for row in robustness['comparison_table']:lines.append(f"| {row['family']} | {row['scenario']} | {row['original_pinball']:.6f} | {row['perturbed_pinball']:.6f} |")
    if 'auxiliary_capacity_qualification.json' in documents:
        capacity=documents['auxiliary_capacity_qualification.json']
        lines.extend(['','### Same-information capacity controls','',
            'Actual trained concat controls match their earlier-inner-selected RGMF reference within 1% parameters. All three seeds and the common target normalizer remain fixed.',
            '| Control | Dates | RGMF pinball | Capacity-matched concat pinball |','|---|---:|---:|---:|'])
        for row in capacity['comparison_table']:lines.append(f"| {row['family']} | {row['n_dates']} | {row['reference_RGMF_pinball']:.6f} | {row['capacity_concat_pinball']:.6f} |")
        lines.append('Paired intervals, actual parameter counts and raw reload deltas are in auxiliary_capacity_qualification.json. These are exploratory Tier B diagnostics.')
    if 'research_figures.json' in documents:
        figures=documents['research_figures.json']
        lines.extend(['','### Standalone development figures','',f"Captured analysis SHA256: `{figures['source_sha256']}`. Figures use pre-reserved development outcomes. [research_figures.json]",
            f"![Development scores and interval coverage]({figures.get('repo_snapshot_directory','reports/figures').removeprefix('reports/')}/development_scores_and_coverage.png)",
            f"![Paired development intervals]({figures.get('repo_snapshot_directory','reports/figures').removeprefix('reports/')}/development_paired_intervals.png)",
            f"![Next-release inventory forecasts]({figures.get('repo_snapshot_directory','reports/figures').removeprefix('reports/')}/inventory_forecasts.png)"])
    for filename in ['auxiliary_cpu_temporal_transport.json','auxiliary_all_family_temporal_transport.json']:
        if filename not in documents:continue
        transport=documents[filename]
        lines.extend(['','### Fixed temporal transport diagnostic','',
            'Same earlier-inner 2022 recipe: frozen 2021-trained model versus 2022-trained refit, scored on the same 2023 targets and common pre-2022 scale. No new model/seed selection.',
            '| Family | Forecast dates | Mature targets | Zero-shot pinball | Refit pinball |','|---|---:|---:|---:|---:|'])
        for row in transport['comparison_table']:lines.append(f"| {row['family']} | {row['forecast_dates']} | {row['scored_dates']} | {row['zero_shot_pinball']:.6f} | {row['refit_pinball']:.6f} |")
        lines.append('Asset-group transport remains blocked: only one qualified physical target entity; Main/Nested source and price gates remain absent. ['+filename+']')
    if 'frozen_gate_controls.json' in documents:
        gates=documents['frozen_gate_controls.json']
        lines.extend(['','### Frozen gate sensitivity','',
            'Stored models are evaluated with uniform valid-source gates and their joint-trained base branch. No refitting occurred; these are neither retrained ablations nor causal source effects.',
            '| Family | Intervention | Dates | Reference pinball | Changed pinball |','|---|---|---:|---:|---:|'])
        for row in gates['comparison_table']:lines.append(f"| {row['family']} | {row['case']} | {row['n_dates']} | {row['reference_pinball']:.6f} | {row['intervention_pinball']:.6f} |")
        lines.append('All raw model receipts and paired date intervals are retained. [frozen_gate_controls.json]')
    if 'track_input_qualification.json' in documents:
        lines.extend(['','### Independent track input gates',''])
        for row in documents['track_input_qualification.json']['tracks']:
            cpu=row.get('cpu_development',{})
            cpu_state=('; CPU '+cpu.get('state','UNKNOWN')) if cpu else ''
            lines.append(f"- {row['track']}: {row['state']}{cpu_state}; {row.get('reason','explicit '+row.get('price_mode','unknown')+' protocol')}. [track_input_qualification.json]")
    if 'experiment_execution_status.json' in documents:
        lines.extend(['','### Experiment completion vector','',
            '| Experiment | Branch state | Remaining boundary |','|---|---|---|'])
        for row in documents['experiment_execution_status.json']['experiments']:
            lines.append(f"| {row['id']} | {row['state']} | {row['reason']} |")
        lines.append('These are independent branch states, not full research PASS assertions. [experiment_execution_status.json]')
    if 'real_systems_qualification.json' in documents:
        systems=documents['real_systems_qualification.json'];probes=systems['probes']
        lines.extend(['','### Actual development inference systems','',
            f"Fixed {systems['protocol']['n_distinct_dates']}-date MLP/2022/seed11 model; no retraining or model selection. Repeated passes measure timing and do not create independent dates.",
            f"Normalized BF16 maximum prediction delta {probes['precision']['max_normalized_abs_delta']:.6g}; quality gate {probes['precision']['quality_passed']}; compile cold/warm timings and failure states retained. No optimization adopted. [real_systems_qualification.json]"])
    if 'fixed_gate_retrained.json' in documents:
        d=documents['fixed_gate_retrained.json']
        lines.extend(['','### Retrained fixed-gate component control','',
            'Same reference earlier-inner-selected widths/LRs, all three seeds and 100 optimizer epochs; no new HPO or scientific promotion. This is separate from the stored-model uniform-gate intervention.',
            '| Family | Reference pinball | Retrained fixed gate | Unique dates |',
            '|---|---:|---:|---:|'])
        for row in d['comparison_table']:lines.append(f"| {row['family']} | {row['reference_pinball']:.6f} | {row['fixed_gate_pinball']:.6f} | {row['n_unique_dates']} |")
        lines.append('Raw reload checks, parameter counts, exact reference receipts and paired intervals retained. [fixed_gate_retrained.json]')
    if 'real_training_systems_qualification.json' in documents:
        actual=documents['real_training_systems_qualification.json'];probes=actual['probes']
        lines.extend(['','### Actual development training systems','',
            f"State: {actual['state']}; {actual['protocol']['n_mature_training_dates']} distinct mature training dates, 100 epochs per fixed MLP replica. Replicas and exposures are not new market samples. No optimization adopted.",
            '| Probe | Recorded quality gate | Maximum output delta | Seconds including cold |','|---|---|---:|---:|'])
        for name in ['microbatch32','microbatch128','compile','recovery','CPU_serial','CPU_overlap']:
            probe=probes[name]
            lines.append(f"| {name} | {probe.get('passed',probe.get('state'))} | {probe.get('max_abs_delta','unavailable')} | {probe.get('seconds_including_cold','unavailable')} |")
        precision=probes['training_precision']
        lines.append(f"Counterbalanced FP32/BF16: {len(precision)} replicas; {sum(bool(p['quality_passed']) for p in precision)} passed the fixed quality gate. Per-replica loss, normalized delta, exposure count and timing retained. [real_training_systems_qualification.json]")
        lines.append('Worker 0/2/4 startup plus pinned copy and cached/uncached training transforms use the same real rows. Recovery is an optimizer-state disk roundtrip, not an OS shutdown. Failures remain diagnostic failures; qualified all-track systems adoption remains blocked.')
    if 'process_recovery_qualification.json' in documents:
        recovery=documents['process_recovery_qualification.json']
        lines.append(f"Actual project-owned CUDA process termination/resume: step {recovery['interrupted_at_step']} to {recovery['final_steps']}, maximum parameter delta {recovery['max_parameter_abs_delta']}. Same-stack synthetic recovery evidence, not host-shutdown recovery. [process_recovery_qualification.json]")
    if 'real_process_recovery_qualification.json' in documents:
        recovery=documents['real_process_recovery_qualification.json']
        lines.extend(['','### Actual development process recovery','',f"State: {recovery['state']}. [real_process_recovery_qualification.json]"])
        if recovery['state']=='PASSED_ACTUAL_DEVELOPMENT_OWN_PROCESS_RECOVERY':
            lines.append(f"Own actual CUDA MLP worker was SIGTERM interrupted after committed step {recovery['interrupted_step']} and resumed to {recovery['final_steps']}. Parameter/prediction maximum deltas {recovery['max_parameter_abs_delta']}/{recovery['max_prediction_abs_delta']}; identical loss trace. Independent training/test dates {recovery['n_distinct_training_dates']}/{recovery['n_distinct_test_dates']}. This is one development workload, not OS shutdown or driver recovery.")
        else:lines.append('Failure retained: '+recovery['reason'])
    if 'systems_qualification.json' in documents:
        s=documents['systems_qualification.json'];p=s['probes']
        lines.extend(['','### Synthetic CUDA capability systems','',f"Measured device: {s['device']}; total {s['total_bytes']} bytes; initial free {s['initial_free_bytes']} bytes. [systems_qualification.json]",
                      f"CPU/CUDA FP32 maximum prediction delta: {p['S1']['max_prediction_abs_delta']:.3g}. [systems_qualification.json:S1]",
                      f"Timed 100-call FP32/BF16 paths: {p['S2']['fp32_seconds']:.6f}/{p['S2']['bf16_seconds']:.6f} seconds; maximum absolute delta {p['S2']['max_abs_delta']:.6g}. BF16 not adopted. Initial autocast overhead is included; this is a small synthetic path, not a general throughput verdict. [systems_qualification.json:S2]",
                      f"Compile first call {p['S3'].get('cold_seconds','unavailable')} seconds; warm 100 calls {p['S3'].get('warm_100_seconds','unavailable')} seconds. Cold cost prevents adoption for this segment. [systems_qualification.json:S3]",
                      f"Microbatch/full-batch maximum gradient delta {p['S4']['max_gradient_abs_delta']:.3g}. [systems_qualification.json:S4]",
                      f"Checkpoint reload parameter delta {p['S8']['max_parameter_abs_delta']}; actual own-process termination/resume evidence is reported separately. [systems_qualification.json:S8]",
                      f"CPU/CUDA FP64 fixed-index statistics maximum delta {p['S10']['max_abs_delta']:.3g}; CUDA took longer on this small probe. [systems_qualification.json:S10]",
                      'These capability probes use synthetic fixtures. Actual development workload experiments are reported separately. Loader startup and one-shot measurements do not justify broad operational defaults. No systems experiment uses reserved data.',
                      'Only one physical GPU was used. No AWS or distributed-scale result exists.'])
    if 'fresh_cpu_reproduction.json' in documents:
        reproduction=documents['fresh_cpu_reproduction.json']
        lines.extend(['','### Fresh package environment reproduction','',
            reproduction['state']+'. [fresh_cpu_reproduction.json]'])
        if reproduction['state']=='PASSED_FRESH_PACKAGE_ENVIRONMENT_TIER_B_CPU_REPRODUCTION':
            lines.append(f"Reproduced {len(reproduction['checks'])} actual CPU outer bundles and their scores with pinned dependencies in fresh isolated site-packages; zero new fits and zero CUDA allocations. The configured WSL interpreter was reused. {reproduction['claim_boundary']}. Package archive hashes and installation log are preserved in the immutable reproduction bundle.")
        else:lines.append('Failure retained: '+reproduction['reason'])
    if 'eia_2023_calibration_source_audit.json' in documents:
        source_audit=documents['eia_2023_calibration_source_audit.json']
        lines.extend(['','### Unresolved calibration source consistency','',
            'The 2023-12-28 official CSV and visually inspected same-release PDF both fail the registered commercial stocks arithmetic check. No PDF substitution was admitted and no model inputs changed. Adjustment/revision cause remains unknown. ['+source_audit['ADR']+']',
            'Exact raw hashes, extracted values, tolerances, parser failures and the image hash are preserved in eia_2023_calibration_source_audit.json. Complete calibration remains blocked.'])
    if 'keyed_source_access.json' in documents:
        lines.extend(['','### Explicit keyed source request boundary','',
            'The FRED adapter requires a registered development-only request plan, explicit observation/vintage bounds, unchanged pagination counts and conservative date-only clocks. Credentials do not qualify source timing or units. Weekly price API history has no server-side end-date filter; it is blocked before downloading reserved history. [keyed_source_access.json]'])
        for source in documents['keyed_source_access.json']['sources']:
            lines.append(source['source']+': '+source['state']+'; '+source.get('reason','raw result remains research-unqualified'))
    if 'tiingo_development_acquisition.json' in documents:
        acquisition=documents['tiingo_development_acquisition.json']
        lines.extend(['','### Bounded Tiingo development P0','',
            'Acquisition state: '+acquisition['state']+'. [tiingo_development_acquisition.json]',
            'Only SPY/QQQ/IEF/TLT/GLD/SLV/USO/UNG, 2010-07-20 through 2023-12-31. Raw close is the unadjusted P0 target basis; adjusted fields are retained separately. EOD availability is conservatively reconstructed as 20:30 America/New_York on the session date, so a Friday 18:00 origin cannot use same-Friday final EOD. This is not authenticated provider publication. Tier B, original-publication-unqualified, development only; no economics or final qualification. [docs/TIINGO_PRICE_PROTOCOL.md]'])
    if 'tiingo_action_acquisition.json' in documents:
        actions=documents['tiingo_action_acquisition.json']
        lines.append('Tiingo corporate-action preparation: '+actions['state']+'; payment-date completeness '+str(actions.get('complete_payment_dates'))+'. Raw action acquisition does not qualify P1 economics. [tiingo_action_acquisition.json]')
    if 'tiingo_input_build.json' in documents:
        lines.append('Tiingo input builder: '+documents['tiingo_input_build.json']['state']+
            '; full Friday origin/asset grid retained, absent information sets reported separately. P1 action/open/pay-date qualification remains blocked. [tiingo_input_build.json]')
    if 'fred_segmented_plan.json' in documents:
        plan=documents['fred_segmented_plan.json']
        lines.extend(['','### v3.1.1 source-chain correction','',
            'FRED transport plan: '+plan['state']+'; '+str(plan['segment_count'])+' explicit segments across '+str(plan['series_count'])+' preregistered series, each revision segment bounded below the provider JSON vintage-date ceiling. No numeric outcome values were used to choose boundaries. [fred_segmented_plan.json]'])
        if 'macro_canonical_integration.json' in documents:
            macro=documents['macro_canonical_integration.json']
            lines.append('Macro integration: '+macro['state']+'; '+str(macro.get('event_rows'))+' canonical Tier-B events. INDPRO uses same-vintage YoY percent to avoid historical index-base mixing. [macro_canonical_integration.json]')
        if 'nport_bulk_integration_v311.json' in documents:
            nport=documents['nport_bulk_integration_v311.json']
            lines.append('N-PORT v3.1.1 integration: '+nport['state']+'; '+str(nport.get('event_rows'))+' canonical target-fund events. The conservative development clock is 23:59:59 America/New_York on the next U.S. federal business day after EDGAR Accepted; Tier B and original-publication-unqualified. [nport_bulk_integration_v311.json]')
        elif 'nport_bulk_integration.json' in documents:
            nport=documents['nport_bulk_integration.json']
            lines.append('N-PORT parent integration: '+nport['state']+'; '+str(nport.get('event_rows'))+' canonical target-fund events. Parent clock assumptions remain provenance only for v3.1.1 source admission. [nport_bulk_integration.json]')
        if 'track_source_input_extension_v311.json' in documents:
            ext=documents['track_source_input_extension_v311.json']
            lines.append('Versioned source inputs: '+ext['state']+'; available sources '+', '.join(ext.get('available_sources',[]))+'. I0 parity is required before extension admission. [track_source_input_extension_v311.json]')
    if 'completion_extension_v31.json' in documents:
        extension=documents['completion_extension_v31.json']
        from .completion import successor_boundary
        live_boundary=successor_boundary(repo)
        track_text=', '.join(k+'='+v['state'] for k,v in live_boundary.get('track_boundaries',{}).items())
        lines.extend(['','### Preregistered Main/Nested completion extension','',
            'Protocol '+extension['protocol_id']+'; live successor boundary '+live_boundary['state']+' ('+track_text+'). Parent v3 pilot and compute ledgers are immutable and are not reset. [completion_extension_v31.json]'])
        if 'track_calibration_status.json' in documents:
            calibration=documents['track_calibration_status.json']
            lines.append('Track calibration gates: '+', '.join(k+'='+v['state'] for k,v in calibration.items())+'. Auxiliary calibration blocking does not automatically block Main/Nested development. [track_calibration_status.json]')
    lines.extend(['','### Successor development continuation',''])
    for name in ['successor_execution_plan.json','successor_gpu_pilot_Main_A.json','successor_gpu_pilot_Nested_B.json',
                 'successor_gpu_bill_admission.json','successor_gpu_full.json','calibration_Main_A.json','calibration_Nested_B.json',
                 'track_development_analysis.json','strict_pit_final_gate_audit.json','successor_continuation_terminal.json']:
        if name in documents:
            doc=documents[name];lines.append(name+': '+doc.get('state','RECORDED')+'. ['+name+']')
            if name=='successor_gpu_bill_admission.json':
                lines.append('Outcome-blind admitted plan: '+str(doc.get('admitted_plan'))+'; immutable full ceiling 21600 GPU seconds. Predictive pilot scores did not enter admission.')
    lines.append('Reconstructed Tier-B P0 development, measured cost pilots and in-sample calibration diagnostics never establish strict-PIT final qualification or P1 economics.')
    lines.extend(['','## Reproduction and resume','',
                  '```powershell',
                  ".\\scripts\\Invoke-SignalForge.ps1 -Action cli -Rest @('reconcile','--read-only')",
                  ".\\scripts\\Invoke-SignalForge.ps1 -Action python -Rest @('scripts/run_validation.py')",
                  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\Resume-SourceChainV311.ps1",
                  ".\\scripts\\Invoke-SignalForge.ps1 -Action python -Rest @('scripts/build_tiingo_inputs.py')",
                  ".\\scripts\\Invoke-SignalForge.ps1 -Action python -Rest @('scripts/run_track_development.py','--track','both','--input-version','auto')",
                  ".\\scripts\\Invoke-SignalForge.ps1 -Action cli -Rest @('report','--study','sgqx-v3','--validate-lineage')",
                  '```','',
                  'FRED/Tiingo/legacy Alpha Vantage resume requires credentials explicitly supplied to the process; SEC requires a genuine contact User-Agent. Exact bounded commands and claim boundaries: docs/TIINGO_PRICE_PROTOCOL.md. No secret locations are searched.',
                  'P1 economics resume requires a permitted raw daily/action panel and a provider/license/clock qualification record. No paid purchase is authorized.',
                  'Full research completion requires qualified all-track source inputs, workload-specific pilot admission and frozen receipts. The completed Auxiliary diagnostics and tested software cannot qualify absent data or bypass the pilot ceiling.',
                  '', '## Observed limitations and next research version','',
                  'Prioritize original publication/vintage evidence, complete archive coverage and qualified prices before enlarging models. Continue v3 development while final is sealed; after any actual unblinding, new scientific choices require a prospective protocol/cohort.',
                  'See future_research_cards.json, model_cards.json and data_cards.json for claim boundaries.'])
    markdown='\n'.join(lines)+'\n'
    atomic_bytes(folder/'technical_report.md',markdown.encode('utf-8'))
    atomic_bytes(folder/'technical_report.html',('<!doctype html><meta charset="utf-8"><title>SignalForge-QX evidence</title><pre>'+html.escape(markdown)+'</pre>').encode('utf-8'))
    atomic_json(folder/'evidence_index.json',{'code_hash':implementation,'captured_at':now(),'snapshot_id':snapshot_id,
                                            'snapshot_receipt_id':digest(receipt),'supporting_artifacts':evidence,
                                            'report_sha256':file_hash(folder/'technical_report.md'),'complete_research_report':False})
    return evidence


def validate_lineage(repo,require_current=False):
    folder=Path(repo)/'reports';index=json.loads((folder/'evidence_index.json').read_text())
    if file_hash(folder/'technical_report.md')!=index['report_sha256']:
        raise ValueError('Report changed outside deterministic generator')
    for evidence in index['supporting_artifacts'].values():
        if file_hash(evidence['path'])!=evidence['sha256']:
            raise ValueError('Supporting artifact changed; regenerate report')
        if require_current and file_hash(evidence.get('live_source_path',evidence['path']))!=evidence['sha256']:
            raise ValueError('Live evidence advanced after report snapshot; regenerate for latest checkpoint')
    return True
