import copy
import pytest
from signalforge.successor_queue import completed_full_receipt


def fixture():
    plan={'plans':{'fixed':{'planned_fit_count':2}},'tracks':['Main-A','Nested-B']}
    admission={'state':'ADMITTED_MEASURED_SUCCESSOR_BILL','execution_plan_id':'frozen','source_tree_hash':'historical-execution-source'}
    full={'state':'SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT','execution_plan_id':'frozen','plan_mode':'fixed','planned_fits':2,'expected_outer_results':2,'reserved_access':False,
          'tracks':[{'track':'Main-A','results':[{'state':'SUCCEEDED','run_id':'a'}]},{'track':'Nested-B','results':[{'state':'SUCCEEDED','run_id':'b'}]}]}
    return full,admission,plan

def test_completed_study_does_not_require_readmission_for_posthoc_source_hash():
    full,admission,plan=fixture()
    assert completed_full_receipt(full,admission,plan,'frozen')
    assert admission['source_tree_hash']=='historical-execution-source'
    # Completion does not set qualification flags or modify the old admission.
    assert 'qualified_for_final' not in full

def test_completed_receipt_integrity_failure_never_dispatches_repeated_work():
    full,admission,plan=fixture()
    for mutate in [lambda r:r.update(planned_fits=3),lambda r:r.update(execution_plan_id='changed'),lambda r:r.update(reserved_access=True),lambda r:r['tracks'][0]['results'][0].update(state='FAILED'),lambda r:r['tracks'][1]['results'][0].update(run_id='a')]:
        broken=copy.deepcopy(full);mutate(broken)
        with pytest.raises(PermissionError):completed_full_receipt(broken,admission,plan,'frozen')

def test_missing_or_blocked_study_is_not_promoted_to_completed():
    full,admission,plan=fixture()
    assert not completed_full_receipt(None,admission,plan,'frozen')
    assert not completed_full_receipt({**full,'state':'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE'},admission,plan,'frozen')

@pytest.mark.parametrize('validated,expected_action',[(True,'IDLE_AFTER_PILOT'),(False,'RUN_VALIDATION')])
def test_actual_queue_prioritizes_current_validation_then_terminal_receipt(tmp_path,monkeypatch,capsys,validated,expected_action):
    import json,runpy
    from pathlib import Path
    import signalforge.runtime as runtime
    import signalforge.completion as completion
    import signalforge.successor as successor
    full,admission,plan=fixture();repo=tmp_path/'repo';(repo/'reports').mkdir(parents=True)
    for name,receipt in [('successor_gpu_full.json',full),('successor_gpu_bill_admission.json',admission)]:
        (repo/'reports'/name).write_text(json.dumps(receipt))
    monkeypatch.setattr(runtime,'paths',lambda:(repo,tmp_path/'runtime'))
    monkeypatch.setattr(runtime,'code_hash',lambda p:'new-posthoc-source')
    monkeypatch.setattr(completion,'successor_boundary',lambda p:{'track_boundaries':{}})
    monkeypatch.setattr(completion,'current_full_validation',lambda p:{'passed':validated,'current_tree':validated})
    monkeypatch.setattr(successor,'successor_authorized',lambda p:True)
    monkeypatch.setattr(successor,'load_frozen_plan',lambda p,r:(plan,{'plan_id':'frozen'}))
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/gpu_queue_status.py'))
    assert json.loads(capsys.readouterr().out)['action']==expected_action
    assert json.loads((repo/'reports/successor_gpu_bill_admission.json').read_text())==admission
