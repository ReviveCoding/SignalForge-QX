import json
from signalforge.track_status import continuation_phase_state

def test_terminal_receipt_does_not_leave_idle_phases_running_or_promote_final(tmp_path):
    reports=tmp_path/'reports';reports.mkdir()
    assert continuation_phase_state(tmp_path,'P08','RUNNING','pending')==('RUNNING','pending')
    (reports/'successor_continuation_terminal.json').write_text(json.dumps({'branches':{'cpu_development':'COMPLETED'}}))
    assert continuation_phase_state(tmp_path,'P07','RUNNING','pending')[0]=='COMPLETED_DEVELOPMENT_TIER_B'
    assert continuation_phase_state(tmp_path,'P08','RUNNING','pending')[0]=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'
    assert continuation_phase_state(tmp_path,'P12','RUNNING','pending')[0]=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'
    assert continuation_phase_state(tmp_path,'P14','RUNNING','pending')[0]=='COMPLETED_INCREMENTAL_DEVELOPMENT_REPORT'


def test_completed_fixed_gpu_receipt_supersedes_historical_runtime_block_only_for_development(tmp_path):
    reports=tmp_path/'reports';reports.mkdir()
    (reports/'successor_continuation_terminal.json').write_text(json.dumps({'branches':{'cpu_development':'COMPLETED'}}))
    full={'state':'SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT','plan_mode':'fixed_setting_development',
          'completed_fit_calls':540,'expected_outer_results':540,'reserved_access':False,'qualified_for_final':False,
          'tracks':[{'track':track,'state':'SUCCEEDED_DEVELOPMENT_SOFTWARE','blocked_folds':[],
                     'results':[{'state':'SUCCEEDED'}]*count} for track,count in [('Main-A',360),('Nested-B',180)]]}
    (reports/'successor_gpu_full.json').write_text(json.dumps(full))
    assert continuation_phase_state(tmp_path,'P08','RUNNING','pending')[0]=='COMPLETED_FIXED_SETTING_DEVELOPMENT_TIER_B'
    assert continuation_phase_state(tmp_path,'P09','RUNNING','pending')[0]=='COMPLETED_FIXED_GRID_WITH_REGISTERED_GAPS'
    for phase in ['P12','P13','P15']:
        assert continuation_phase_state(tmp_path,phase,'RUNNING','pending')[0]=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'
    full['tracks'][1]['results'].pop()
    (reports/'successor_gpu_full.json').write_text(json.dumps(full))
    assert continuation_phase_state(tmp_path,'P08','RUNNING','pending')[0]=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'


def test_track_calibration_completion_requires_both_receipts_without_promoting_final(tmp_path):
    reports=tmp_path/'reports';reports.mkdir()
    (reports/'successor_continuation_terminal.json').write_text('{}')
    for track in ['Main_A','Nested_B']:
        (reports/('calibration_'+track+'.json')).write_text(json.dumps({'state':'COMPLETED_DEVELOPMENT_CALIBRATION_TIER_B','candidates':[{}],'qualified_for_final':False}))
    assert continuation_phase_state(tmp_path,'P10','RUNNING','pending')[0]=='COMPLETED_DEVELOPMENT_CALIBRATION_TIER_B'
    (reports/'calibration_Nested_B.json').unlink()
    assert continuation_phase_state(tmp_path,'P10','RUNNING','pending')[0]=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'
