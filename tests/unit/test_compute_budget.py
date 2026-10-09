import pytest
from signalforge.experiments import ComputeBudget
from signalforge.runtime import commit_bundle, validate_bundle


def test_consumption_persists_and_failure_charged(tmp_path):
    path=tmp_path/'budget.sqlite';budget=ComputeBudget(path,100)
    with budget.charge('one',10):pass
    with pytest.raises(ValueError):
        with budget.charge('failure',10):raise ValueError('fit fails')
    left=budget.remaining
    assert left<100
    budget.close();budget=ComputeBudget(path,100)
    assert budget.remaining==left
    with pytest.raises(ValueError):
        with budget.charge('one',1):pass
    budget.close()
    with pytest.raises(PermissionError):ComputeBudget(path,200)


def test_crash_reservation_and_atomic_admission(tmp_path):
    budget=ComputeBudget(tmp_path/'budget.sqlite',100)
    with budget.db:budget.db.execute('INSERT INTO compute_charges VALUES (?,?,?)',('killed',90,'RESERVED'))
    assert budget.remaining==10
    with pytest.raises(RuntimeError,match='PAUSED_BUDGET'):
        with budget.charge('new',11):pass
    assert budget.remaining==10
    budget.close()


def test_binary_model_bundle_checksum(tmp_path):
    commit_bundle(tmp_path,{'model.pt':b'local-model-fixture','metadata.json':{'x':1}}, {'study':'fixture'})
    assert 'model.pt' in validate_bundle(tmp_path)['artifacts']
    (tmp_path/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):validate_bundle(tmp_path)
