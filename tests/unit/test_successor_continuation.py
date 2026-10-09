import json
from pathlib import Path
import pytest
from signalforge.runtime import file_hash
from signalforge.successor import successor_authorized,require_successor_access,execution_plan,measured_admission,classify_failure

ROOT=Path(__file__).parents[2]

def repo_fixture(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'.local').mkdir()
    for name in ['completion_extension_v31.json','local_rtx4090_laptop.json','study.json','statistical_contract.json']:
        (tmp_path/'configs'/name).write_bytes((ROOT/'configs'/name).read_bytes())
    return tmp_path

def auth_fixture(repo):
    auth={'authorized':True,'scope':'registered_successor_gpu_development_only','protocol_id':'sgqx-v3.1-track-gpu',
          'protocol_sha256':file_hash(repo/'configs/completion_extension_v31.json'),'reserved_access':False,
          'budget_increase_authorized':False,'pilot_fit_ceiling':60,'pilot_gpu_seconds':3600,'full_gpu_seconds':21600}
    (repo/'.local/ALLOW_SUCCESSOR_GPU').write_text(json.dumps(auth));return auth

def test_transition_requires_exact_development_scope_and_preserves_pause(tmp_path):
    repo=repo_fixture(tmp_path)
    assert not successor_authorized(repo)
    auth_fixture(repo);assert successor_authorized(repo)
    (repo/'.local/PAUSE_BEFORE_GPU').write_text('manual pause')
    with pytest.raises(PermissionError,match='PAUSE_BEFORE_GPU'):require_successor_access(repo)
    assert (repo/'.local/PAUSE_BEFORE_GPU').exists()

@pytest.mark.parametrize('field,value',[('reserved_access',True),('full_gpu_seconds',21601),('scope','final'),('protocol_sha256','bad')])
def test_authorization_never_grants_reserved_or_budget_extension(tmp_path,field,value):
    repo=repo_fixture(tmp_path);auth=auth_fixture(repo);auth[field]=value
    (repo/'.local/ALLOW_SUCCESSOR_GPU').write_text(json.dumps(auth))
    with pytest.raises(PermissionError):successor_authorized(repo)

def test_registered_plan_count_and_outcome_blind_fallback(tmp_path):
    plan=execution_plan(repo_fixture(tmp_path))
    assert plan['plans']['registered_hpo']['planned_fit_count']==4140
    assert plan['plans']['fixed_setting_development']['planned_fit_count']==540
    assert plan['seeds']==[11,37,71]
    costs={track:{f:1. for f in plan['families']} for track in plan['tracks']}
    admitted=measured_admission(plan,costs)
    assert admitted['admitted_plan']=='fixed_setting_development' # 16x width margin disallows full HPO.
    costs={track:{f:100. for f in plan['families']} for track in plan['tracks']}
    assert measured_admission(plan,costs)['state']=='RESOURCE_OR_BUDGET_GATE'
    assert plan['full_gpu_seconds']==21600

def test_nonfinite_measurement_cannot_admit(tmp_path):
    plan=execution_plan(repo_fixture(tmp_path));costs={t:{f:float('nan') for f in plan['families']} for t in plan['tracks']}
    with pytest.raises(ValueError):measured_admission(plan,costs)

@pytest.mark.parametrize('message,expected',[
    ('checksum mismatch','INTEGRITY_OR_HASH_GATE'),('PAUSED_BUDGET','RESOURCE_OR_BUDGET_GATE'),
    ('reserved authorization','AUTH_GATE'),('BLOCKED_DATA maturity','SCIENTIFIC_OR_DATA_GATE'),
    ('CUDA lease held','RESOURCE_CONTENTION'),('CUDA error','CUDA_RUNTIME_OR_LIBRARY_DEFECT'),
    ('AttributeError','CODE_ORCHESTRATION_DEFECT'),('something unfamiliar','UNKNOWN_UNSAFE')])
def test_failure_classes_fail_closed(message,expected):
    assert classify_failure(message)==expected

def test_powershell_recovery_authorization_precedes_pause_creation():
    text=(ROOT/'scripts/Watch-SignalForgeRecovery.ps1').read_text()
    body=text.split('function Ensure-GpuPause {',1)[1].split('function Get-ArtifactSummary',1)[0]
    assert body.index('if(Test-SuccessorGpuAuthorization){return}')<body.index('Set-Content')
    assert "if($MaxRepairAttempts -ne 2)" in text

def test_conditional_support_uses_dates_not_assets_or_seeds():
    import pandas as pd
    from signalforge.successor import conditional_family_gate
    grid=pd.DataFrame([{'decision_time':d,'asset':a,'y':1.,'context_eligible':True} for d in range(599) for a in range(8)])
    gate={'minimum_independent_mature_decision_dates':600,'minimum_assets_with_target_coverage':4}
    assert conditional_family_gate(grid,gate)['state']=='NOT_ADMITTED_CONDITIONAL_FAMILY'
    grid=pd.concat([grid,pd.DataFrame([{'decision_time':599,'asset':a,'y':1.,'context_eligible':True} for a in range(4)])])
    assert conditional_family_gate(grid,gate)['state']=='ADMITTED_CONDITIONAL_FAMILY'

def test_frozen_plan_detects_materialized_config_mutation(tmp_path):
    from signalforge.successor import freeze_execution_plan,load_frozen_plan
    (tmp_path/'repo').mkdir();repo=repo_fixture(tmp_path/'repo')
    runtime=tmp_path/'runtime';runtime.mkdir()
    config=repo/'configs/successor_full_study_v31.json'
    config.write_text(json.dumps(execution_plan(repo)))
    freeze_execution_plan(repo,runtime);load_frozen_plan(repo,runtime)
    changed=json.loads(config.read_text());changed['seeds']=[11]
    config.write_text(json.dumps(changed))
    with pytest.raises(PermissionError,match='materialized'):load_frozen_plan(repo,runtime)

def test_cached_panel_guard_rejects_reserved_origin_before_target_open(tmp_path):
    from signalforge.successor_inputs import prepared_track
    (tmp_path/'repo').mkdir();repo=repo_fixture(tmp_path/'repo')
    runtime=tmp_path/'runtime';runtime.mkdir()
    calendar=runtime/'calendar.json';calendar.write_text('["2024-01-05T23:00:00Z"]')
    manifest={'reserved_access':False,'decisions':{'relative_path':calendar.name,'sha256':file_hash(calendar)},
              'universe_card':{'assets':['SPY'],'outcome_blind':True,'frozen_before_outcomes':'fixture'}}
    (repo/'.local/inputs_Main-A_v311.json').write_text(json.dumps(manifest))
    with pytest.raises(PermissionError,match='reserved'):prepared_track(repo,runtime,'Main-A')

def test_phase_status_uses_successful_cpu_receipts_without_promoting_final(tmp_path):
    from signalforge.track_status import track_phase_states
    (tmp_path/'reports').mkdir()
    q={'tracks':[{'track':t,'state':'READY_RECONSTRUCTED_DEVELOPMENT_INPUTS','cpu_development':{'state':'SUCCEEDED_DEVELOPMENT_SOFTWARE'}} for t in ['Main-A','Nested-B']]}
    (tmp_path/'reports/track_input_qualification_v311.json').write_text(json.dumps(q))
    assert track_phase_states(tmp_path,'P07')['Main-A']=='SUCCEEDED_DEVELOPMENT_SOFTWARE'
    assert track_phase_states(tmp_path,'P08')['Nested-B']=='CPU_SUCCEEDED_GPU_PENDING'
    assert track_phase_states(tmp_path,'P12')['Main-A']=='BLOCKED_FREEZE_GATES'
    assert track_phase_states(tmp_path,'P15')['Nested-B']=='BLOCKED_OPERATIONAL_FREEZE'
