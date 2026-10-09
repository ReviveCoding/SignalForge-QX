import ast,json,runpy
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pytest
from signalforge.runtime import atomic_json,code_hash,file_hash

ROOT=Path(__file__).resolve().parents[2]

def test_optimizer_failure_returns_registered_feasible_cash(monkeypatch):
    import cvxpy as cp
    from signalforge.portfolio import feasible_weights
    def failed(*args,**kwargs):raise cp.error.SolverError('controlled failure')
    monkeypatch.setattr(cp.Problem,'solve',failed)
    w,state=feasible_weights([.1,.2],.001,np.eye(2),investable_fraction=.3)
    assert state=='CASH_FALLBACK' and np.array_equal(w,[0.,0.]) and w.sum()<=.3

def test_resume_preserves_every_user_edited_desktop_input(tmp_path,monkeypatch):
    import signalforge.runtime as runtime
    monkeypatch.setattr(runtime,'paths',lambda:(tmp_path,tmp_path/'runtime'))
    folder=tmp_path/'research/desktop_study';folder.mkdir(parents=True)
    names=['question_contract.json','literature_matrix.json','data_dictionary.json','source_access_matrix.json','mechanism_cards.json','method_source_decisions.json']
    for name in names:(folder/name).write_text('user curated '+name)
    before={name:(folder/name).read_bytes() for name in names}
    class Handoff(Exception):pass
    def execv(exe,args):
        assert args[-1]==str(tmp_path/'scripts/audit_desktop_study.py')
        raise Handoff
    monkeypatch.setattr('os.execv',execv)
    with pytest.raises(Handoff):runpy.run_path(str(ROOT/'scripts/build_desktop_study.py'))
    assert {name:(folder/name).read_bytes() for name in names}==before

def test_native_launcher_keeps_interactive_workspace_auto_review_contract():
    text=(ROOT/'scripts/Start-InteractiveCodex.ps1').read_text()
    # Inspect the actual argv assignment, not comments or documentation.
    argv=text.split('$argv=@(',1)[1].split(')\n',1)[0]
    assert "'--sandbox','workspace-write'" in argv and "'--ask-for-approval','on-request'" in argv
    assert 'approvals_reviewer="auto_review"' in argv and "'--strict-config'" in argv
    assert "'exec'" not in argv and '--yolo' not in argv and "'never'" not in argv
    assert '$c.codex_js' in argv and '$RepoRoot' in argv
    assert 'if (-not $IsWindows)' in text and '& $c.node @argv' in text

def test_rootless_setup_has_no_global_settings_or_driver_commands():
    import shlex
    text=(ROOT/'scripts/bootstrap_wsl.sh').read_text()
    commands=[line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    for line in commands:
        words=shlex.split(line,comments=True) if not line.rstrip().endswith('\\') else line.split()
        assert not any(w in {'sudo','apt','apt-get','modprobe','nvidia-settings'} for w in words)
        assert '/etc/' not in line and '.wslconfig' not in line and 'nvidia-smi -' not in line
    assert '[[ $(id -u) != 0 ]]' in text and 'runtime must be under Linux HOME' in text

def test_forward_once_is_forecast_only_one_idempotent_iteration(tmp_path,monkeypatch):
    import signalforge.forward as forward
    import signalforge.integrity as integrity
    from signalforge.frozen_processing import candidate_ensemble_id
    repo=tmp_path/'repo';runtime=tmp_path/'runtime';repo.mkdir();runtime.mkdir()
    candidate={'candidate_id':'one','track':'Main-A','members':[{'member_id':'11','relative_bundle':'models/one','receipt_id':'a'*64}],'weights':[1.]}
    ensemble_id=candidate_ensemble_id(candidate)
    ensemble={'candidate_ensemble_ids':{'one':ensemble_id}}
    minimum={str(q):52 if q!=.5 else 20 for q in [.05,.1,.5,.9,.95]}
    calibration={'candidates':{'one':{'ensemble_id':ensemble_id,'corrections':[0.]*5,'quantiles':[.05,.1,.5,.9,.95],
        'fit_window':{'start':'2023-07-01T00:00Z','end':'2023-12-31T23:59Z','max_label_available_at':'2023-12-30T00:00Z'},
        'fit_data_id':'b'*64,'support':{str(q):{'n_dates':26,'status':'CORRECTION' if q==.5 else 'IDENTITY_INSUFFICIENT_SUPPORT'} for q in [.05,.1,.5,.9,.95]}}}}
    components={}
    for name,value in [('ensemble',ensemble),('calibration',calibration)]:
        path=repo/(name+'.json');atomic_json(path,value);components[name]={'path':str(path),'sha256':file_hash(path)}
    atomic_json(repo/'configs/statistical_contract.json',{'calibration':{'minimum_distinct_dates':minimum}})
    features={'decision_time':'2026-10-09T22:00Z','expected_outcome_available_at':'2026-10-16T22:00Z',
        'max_dependency_available_at':'2026-10-09T21:00Z','assets':['one'],'x':[[[1.]]],'context_eligible':[True]}
    atomic_json(runtime/'features.json',features)
    freeze={'status':'READY_FOR_FORWARD','qualified_real_development':True,'components':components,'source_code_hash':code_hash(repo),
        'current_feature_snapshot':'features.json','frozen_at':'2026-10-08T22:00Z','candidates':[candidate]}
    atomic_json(repo/'.local/operational_freeze.json',freeze)
    monkeypatch.setattr(integrity,'verify_qualification',lambda *args:None)
    monkeypatch.setattr(forward,'now',lambda:'2026-10-09T22:01Z')
    monkeypatch.setattr(forward,'gpu_lease',lambda *args:nullcontext())
    calls=[]
    def predict(*args,**kwargs):calls.append(1);return {'one':(np.array([.1]),np.array([[-2.,-1.,0.,1.,2.]]))}
    monkeypatch.setattr(forward,'infer_candidates',predict)
    def forbidden(*args,**kwargs):raise AssertionError('Forward must not spawn a daemon, broker or network process')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    first=forward.produce_once(repo,runtime,'NOW');second=forward.produce_once(repo,runtime,'NOW')
    assert first==second and calls==[1] and first['real_orders'] is False and first['observed_outcome'] is None
    assert len(list((runtime/'artifacts/forward').iterdir()))==1

def test_reporting_cannot_promote_capability_or_posthoc_slices_to_scientific_claims(tmp_path):
    from signalforge.reporting import render,validate_lineage
    atomic_json(tmp_path/'IMPLEMENTATION_STATUS.json',{'single_gpu_qualified_on_user_device':True,'reserved_evaluation_complete':False})
    render(tmp_path);assert validate_lineage(tmp_path,require_current=True)
    text=(tmp_path/'reports/technical_report.md').read_text()
    assert 'No predictive superiority, calibrated tail coverage, alpha, economic qualification or causal source P&L attribution is asserted.' in text
    assert '| AWS/physical multi-GPU operation | NOT_EXECUTED |' in text
