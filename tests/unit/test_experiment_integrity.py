import pytest
from signalforge.experiments import TrialLedger,measured_bill,eligible_candidate,validate_architecture_contrast
from signalforge.integrity import REQUIRED_FREEZE_COMPONENTS,REQUIRED_FREEZE_GATES,authorize_final,protocol_hash
from signalforge.runtime import atomic_json,file_hash,code_hash


def test_trial_failures_preserved(tmp_path):
    ledger=TrialLedger(tmp_path/'trials.sqlite')
    one=ledger.register({'partition':'inner_validation','experiment':'E02','seed':11})
    ledger.finish(one,'FAILED_PERMANENT',failure='OOM')
    two=ledger.register({'partition':'inner_validation','experiment':'E02','seed':37})
    ledger.finish(two,'PRUNED')
    assert len(ledger.rows())==2
    with pytest.raises(ValueError):ledger.finish(one,'SUCCEEDED',metric=0)
    ledger.close()


def test_hpo_never_outer_or_final(tmp_path):
    ledger=TrialLedger(tmp_path/'trials.sqlite')
    for partition in ['outer','reserved','calibration']:
        with pytest.raises(PermissionError):ledger.register({'partition':partition})
    with pytest.raises(ValueError):ledger.register({'partition':'inner_validation','experiment':'E04'})
    ledger.close()


def test_invalid_control_never_promoted():
    assert not eligible_candidate('E04','development')
    with pytest.raises(PermissionError):eligible_candidate('E10','reserved')


def test_measured_budget_bill():
    assert measured_bill(10,20,150,1024,2*1024**3)['state']=='PAUSED_BUDGET'
    with pytest.raises(ValueError):measured_bill(None,20,150,1024,2*1024**3)


def test_architecture_same_information():
    d={key:'a' for key in ['track','price_mode','pit_tier','grid_hash','normalizer_id','information_hash','context','horizon']}
    assert validate_architecture_contrast(d,d)
    with pytest.raises(ValueError):validate_architecture_contrast(d,{**d,'information_hash':'b'})


def test_same_frozen_batch_resume_only(tmp_path,monkeypatch):
    # Exercise one-batch policy only. Fixture bytes never qualify source science.
    monkeypatch.setattr('signalforge.integrity.verify_qualification',lambda receipt,repo: None)
    monkeypatch.setattr('signalforge.integrity.verify_frozen_models',lambda receipt,runtime: None)
    auth={'authorized':True,'study_id':'sgqx-v3','scope':'one_registered_frozen_batch_after_all_track_gates'}
    components={}
    (tmp_path/'configs').mkdir();(tmp_path/'reports').mkdir()
    for key in REQUIRED_FREEZE_COMPONENTS:
        p=tmp_path/(key+'.json');atomic_json(p,{'fixture':True})
        components[key]={'path':str(p),'sha256':file_hash(p)}
    environment=tmp_path/'reports/environment_audit.json';atomic_json(environment,{'fixture':True})
    components['environment']={'path':str(environment),'sha256':file_hash(environment)}
    receipt={'study_id':'sgqx-v3','status':'READY_FOR_FINAL','protocol_hash':protocol_hash(tmp_path),'components':components,
             'source_code_hash':code_hash(tmp_path),'economics_state':'BLOCKED_PRICE_P1',
             'registered_candidate_ids':['fixture_base','fixture_candidate'],'registered_contrast_ids':['fixture_contrast'],
             'gates':{g:{'state':'SUCCEEDED','evidence_kind':'qualified_real_development','evidence_components':['models']} for g in REQUIRED_FREEZE_GATES}}
    first=authorize_final(receipt,tmp_path,tmp_path,auth,'locked')
    assert authorize_final(receipt,tmp_path,tmp_path,auth,'locked')==first
    changed={**receipt,'protocol_hash':'different'}
    with pytest.raises(PermissionError):authorize_final(changed,tmp_path,tmp_path,auth,'locked')
    p=tmp_path/'models.json';atomic_json(p,{'changed':True})
    with pytest.raises(PermissionError):authorize_final(receipt,tmp_path,tmp_path,auth,'locked')


def test_hash_only_receipt_never_qualifies(tmp_path):
    from signalforge.integrity import verify_qualification
    with pytest.raises(PermissionError,match='gates incomplete'):
        verify_qualification({'components':{},'gates':{}},tmp_path)
    receipt={'components':{'models':{}},'gates':{g:{'state':'SUCCEEDED','evidence_kind':'synthetic_fixture','evidence_components':['models']}
                                               for g in REQUIRED_FREEZE_GATES}}
    with pytest.raises(PermissionError,match='Unqualified scientific'):
        verify_qualification(receipt,tmp_path)


def test_retry_is_new_bounded_attempt_and_keeps_failure(tmp_path):
    ledger=TrialLedger(tmp_path/'trials.sqlite')
    original=ledger.register({'partition':'inner_validation','experiment':'E02','seed':11})
    ledger.finish(original,'INTERRUPTED',failure='Verified external interruption')
    retry=ledger.bounded_retry(original)
    assert retry!=original and ledger.get(original)[2]=='INTERRUPTED'
    assert ledger.bounded_retry(original)==retry
    ledger.finish(retry,'FAILED_RETRYABLE',failure='External GPU owner')
    with pytest.raises(RuntimeError,match='ceiling'):ledger.bounded_retry(original)
    assert len(ledger.rows())==2
    ledger.close()
