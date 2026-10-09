from types import SimpleNamespace
import os,json
import pytest
from signalforge.runtime import gpu_lease


def test_cross_framework_lease_excludes_second_owner(tmp_path,monkeypatch):
    monkeypatch.setattr('signalforge.runtime.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=''))
    with gpu_lease(tmp_path):
        with pytest.raises(RuntimeError,match='CUDA lease held'):
            with gpu_lease(tmp_path):pytest.fail('Second framework acquired the same lease')
        receipt=json.loads((tmp_path/'locks/gpu0.lock').read_text())
        assert receipt['pid']==os.getpid() and receipt['boot_id']
    with gpu_lease(tmp_path):pass


def test_external_cuda_owner_denied(tmp_path,monkeypatch):
    monkeypatch.setattr('signalforge.runtime.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='987654321\n'))
    with pytest.raises(RuntimeError,match='unrelated CUDA owner'):
        with gpu_lease(tmp_path):pytest.fail('External CUDA owner ignored')


def test_unknown_gpu_inventory_denied(tmp_path,monkeypatch):
    monkeypatch.setattr('signalforge.runtime.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    with pytest.raises(RuntimeError,match='unknown inventory'):
        with gpu_lease(tmp_path):pytest.fail('Failed inventory treated as a free GPU')
    monkeypatch.setattr('signalforge.runtime.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='[N/A]\n'))
    with pytest.raises(RuntimeError,match='unknown inventory'):
        with gpu_lease(tmp_path):pytest.fail('Malformed successful inventory treated as free')


def test_repeated_unexplained_shutdown_pauses_and_preserves_incidents(tmp_path):
    from signalforge.runtime import check_shutdown_history
    previous={'state':'RUNNING','pid':100,'boot_id':'boot-a','started_at':'2020-01-01T00:00Z'}
    first=check_shutdown_history(tmp_path,previous,'boot-b')
    assert first['consecutive_unexplained']==1
    assert check_shutdown_history(tmp_path,previous,'boot-b')['consecutive_unexplained']==1
    previous={'state':'RUNNING','pid':101,'boot_id':'boot-b','started_at':'2020-01-02T00:00Z'}
    with pytest.raises(RuntimeError,match='PAUSED_RECOVERY'):check_shutdown_history(tmp_path,previous,'boot-c')
    history=json.loads((tmp_path/'ledger/gpu_shutdown_history.json').read_text())
    assert len(history['incidents'])==2 and history['state']=='PAUSED_RECOVERY'
    with pytest.raises(RuntimeError,match='PAUSED_RECOVERY'):check_shutdown_history(tmp_path,{},'boot-c')


def test_graceful_release_marks_lease(tmp_path,monkeypatch):
    monkeypatch.setattr('signalforge.runtime.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=''))
    with pytest.raises(ValueError):
        with gpu_lease(tmp_path):raise ValueError('Owned process graceful exception')
    assert json.loads((tmp_path/'locks/gpu0.lock').read_text())['state']=='RELEASED'
