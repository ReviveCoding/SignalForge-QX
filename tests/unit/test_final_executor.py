import pytest
from signalforge.final import execute_frozen,read_reserved
from signalforge.runtime import atomic_json


def test_final_executor_rejects_before_reserved_payload_read(tmp_path):
    atomic_json(tmp_path/'.local/scientific_authorization.json',{'authorized':True,'study_id':'sgqx-v3','scope':'one_registered_frozen_batch_after_all_track_gates'})
    receipt=tmp_path/'receipt.json';atomic_json(receipt,{'study_id':'sgqx-v3','status':'READY_FOR_FINAL','components':{}})
    with pytest.raises(PermissionError):execute_frozen(tmp_path,tmp_path,receipt)
    assert not (tmp_path/'ledger/final_access_events.jsonl').exists()


def test_reserved_payload_snapshot_cannot_change(tmp_path):
    path=tmp_path/'targets.json';atomic_json(path,{'fixture_only':True})
    with pytest.raises(PermissionError,match='admission'):read_reserved(path,'labels','fake_freeze',tmp_path,tmp_path)
    atomic_json(tmp_path/'ledger/final_access.json',{'freeze_id':'fake_freeze','fixture':True})
    assert read_reserved(path,'labels','fake_freeze',tmp_path,tmp_path)['fixture_only']
    atomic_json(path,{'fixture_only':False})
    with pytest.raises(PermissionError,match='snapshot changed'):read_reserved(path,'labels','fake_freeze',tmp_path,tmp_path)
