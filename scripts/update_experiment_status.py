"""Separate executed contrasts, independent blockers and conditional method gates."""
import json
import pandas as pd
from signalforge.runtime import paths,atomic_json,now,file_hash
from signalforge.data import auxiliary_rows
repo,runtime=paths()
def evidence(name):
    path=repo/'reports'/name
    if not path.exists():return {'state':'NOT_RUN','path':'reports/'+name}
    value=json.loads(path.read_text())
    return {'state':value.get('state'),'path':'reports/'+name,'sha256':file_hash(path)}
core=evidence('auxiliary_development_results.json')
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
rows=auxiliary_rows(acquisition['records'])
train=rows[pd.to_datetime(rows.label_available_at,utc=True).lt('2023-01-01T00:00Z')&rows.y.notna()]
counts={'independent_training_dates':int(train.decision_time.nunique()),'physical_entities':int(train.asset.nunique()),
    'pseudo_repetition_used':False,'reserved_features_used':False}
cards={
 'E01':{'state':'BLOCKED_DATA','reason':'Main/Nested matched canonical source information ladder absent; bounded P0 I0 alone cannot execute registered I1-I4 contrasts; Auxiliary has one source'},
 'E02':{'state':'RUNNING_OR_PARTIAL','branches':[core,evidence('auxiliary_capacity_qualification.json')],'reason':'Complete raw reload and same-information capacity diagnostics required; Main/Nested remain independently blocked'},
 'E03':{'state':'PARTIAL','branches':[core,evidence('frozen_gate_controls.json'),evidence('fixed_gate_retrained.json')],'reason':'Retrained age/coverage/residual/fixed-gate comparisons are separate from frozen gate/base/outage interventions; all-track qualification remains blocked'},
 'E04':{'state':'PARTIAL','branches':[evidence('auxiliary_invalid_cpu_controls.json'),evidence('auxiliary_invalid_cuda_controls.json')],'reason':'CPU and primary CUDA-tree permutation/wrong-clock/future-target controls are separately retained; latest-revision control blocked by unavailable original vintages','promotion_eligible':False},
 'E05':{'state':'PARTIAL','branches':[evidence('auxiliary_cpu_robustness.json'),evidence('auxiliary_all_family_robustness.json')],'reason':'Frozen input outages are distinct from retrained ablations and post-hoc explanations'},
 'E06':{'state':'PARTIAL','branches':[evidence('auxiliary_cpu_temporal_transport.json'),evidence('auxiliary_all_family_temporal_transport.json')],'reason':'Actual fixed 2023 zero-shot/refit temporal diagnostics; asset-group transport blocked by one available physical entity'},
 'E07':{'state':'DEFERRED_METHOD','reason':'Framework section 9.5 permits inadequate-data deferral. One US physical entity and fewer than 700 independent mature training origins cannot justify 100k-2M parameter scale/weighting transport; additional NPORT rows or repeated windows do not create independent targets.',
        'sample_evidence':counts,'planned_runs':54,'large_models_fitted':0,'core_comparisons_continue':True},
 'E08':{'state':'RUNNING_OR_PARTIAL','branches':[core],'reason':'Small train-cutoff-only RGMF SSL is included in core development; no reserved-period pretraining or large-scale claim'},
 'E09':{'state':'BLOCKED_DATA','reason':'P1 raw-open/action completeness/pay-date/publication-clock audit absent; reconstructed Tiingo P0 and accounting software tests cannot qualify economics'},
 'E10':{'state':'PARTIAL','branches':[evidence('systems_qualification.json'),evidence('real_systems_qualification.json'),evidence('real_training_systems_qualification.json'),evidence('process_recovery_qualification.json'),evidence('real_process_recovery_qualification.json')],'reason':'Synthetic capability checks and fixed actual development inference/training/own-process recovery systems diagnostics are separate; qualified all-track workloads remain absent'}}
if core['state']=='SUCCEEDED_DIAGNOSTIC' and evidence('auxiliary_capacity_qualification.json')['state']=='SUCCEEDED_ACTUAL_CUDA_CAPACITY_RELOAD_AND_COMPARISON':
    cards['E02']['state']='COMPLETED_AUXILIARY_DIAGNOSTIC_ALL_TRACKS_BLOCKED'
if core['state']=='SUCCEEDED_DIAGNOSTIC':cards['E08']['state']='COMPLETED_SMALL_AUXILIARY_SSL_DIAGNOSTIC'
track_path=repo/'reports/track_input_qualification_v311.json'
track_doc=json.loads(track_path.read_text()) if track_path.exists() else {}
track_rows={row.get('track'):row for row in track_doc.get('tracks',[])}
track_cpu_ready=all(
    track_rows.get(track,{}).get('cpu_development',{}).get('state')=='SUCCEEDED_DEVELOPMENT_SOFTWARE' and
    not track_rows.get(track,{}).get('cpu_development',{}).get('blocked_folds')
    for track in ['Main-A','Nested-B']
)
if track_cpu_ready:
    cards['E01'].update(state='PARTIAL',reason='Main/Nested reconstructed fixed-grid CPU information-ladder diagnostics completed; successor GPU architecture work and strict final qualification remain separate')
    if core['state']=='SUCCEEDED_DIAGNOSTIC':
        cards['E02'].update(state='PARTIAL',reason='Auxiliary architecture/capacity diagnostics and Main/Nested CPU baselines are complete; separately admitted successor GPU architecture pilots remain pending')
cards['E01']['branches']=[evidence('track_input_qualification_v311.json'),evidence('track_development_analysis.json'),evidence('successor_gpu_full.json')]
cards['E02']['branches'].extend([evidence('successor_gpu_pilot_Main_A.json'),evidence('successor_gpu_pilot_Nested_B.json'),evidence('successor_gpu_bill_admission.json'),evidence('successor_gpu_full.json')])
if evidence('successor_gpu_full.json')['state']=='SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT':
    cards['E01'].update(state='COMPLETED_DEVELOPMENT_TIER_B',reason='Registered information increments executed on reconstructed development inputs; strict original-publication and final gates remain blocked')
    cards['E02'].update(state='COMPLETED_DEVELOPMENT_TIER_B',reason='Measured-bill-admitted successor family comparison completed under frozen execution plan; no strict final or tuned-family claim beyond admitted design')
cards['E09']['branches']=[evidence('tiingo_input_build.json'),evidence('strict_pit_final_gate_audit.json')]
if evidence('forward_diagnostic_admission_v32.json')['state']!='NOT_RUN':
    for identifier in ['E03','E05','E06','E07','E08','E10']:
        cards[identifier].setdefault('branches',[]).append(evidence('forward_diagnostic_admission_v32.json'))
    cards['E06']['reason']='Auxiliary one-entity asset transport remains blocked; Main/Nested each have eight physical assets, but existing all-asset models are not zero-shot models. Train-group-only preprocessing contract passes; separate future transport fits require original clocks, operational freeze, future cohort and measured admitted bill.'
    cards['E09']['branches'].append(evidence('issuer_action_evidence_2023.json'))
    cards['E09']['reason']='24 issuer IEF/TLT 2023 ex/pay-date records are partial evidence; complete 2010-2023 all-asset actions, raw-open publication clocks, as-of cash benchmark and entitlement remain unqualified. No P0 substitution.'
    cards['E03']['branches'].append(evidence('track_model_forensic_interpretation.json'))
registry=json.loads((repo/'configs/experiments.json').read_text())
result={'created_at':now(),'state':'INCOMPLETE','full_research_complete':False,'registry_sha256':file_hash(repo/'configs/experiments.json'),
    'experiments':[{**entry,**cards[entry['id']],'qualified_for_final':False} for entry in registry['experiments']],
    'reserved_access':False,'economic_qualified':False}
atomic_json(repo/'reports/experiment_execution_status.json',result)
print(json.dumps({entry['id']:entry['state'] for entry in result['experiments']}))
