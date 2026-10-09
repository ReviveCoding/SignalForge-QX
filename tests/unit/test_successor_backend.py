import json
from pathlib import Path
import pytest
from signalforge.successor_backend import activate_backend
from signalforge.successor import classify_failure

def test_native_cuda_abort_is_library_defect():
    assert classify_failure('[LightGBM] [Fatal] [CUDA] an illegal memory access was encountered') == 'CUDA_RUNTIME_OR_LIBRARY_DEFECT'

def test_private_backend_path_must_stay_in_runtime(tmp_path):
    repo=tmp_path/'repo';(repo/'reports').mkdir(parents=True)
    (repo/'reports/successor_cuda_backend.json').write_text(json.dumps({'package_relative_path':'../../outside','version':'4.6.0'}))
    with pytest.raises(PermissionError,match='path/version'):activate_backend(repo,tmp_path/'runtime')

def test_native_failure_with_zero_results_requires_verified_repair():
    from signalforge.successor import verified_pilot_retry
    with pytest.raises(PermissionError,match='verified bounded'):
        verified_pilot_retry(True,[],{'state':'OPEN','repair_attempt':0})
    assert verified_pilot_retry(True,[],{'state':'RESOLVED_VERIFIED','repair_attempt':2})==2
    with pytest.raises(PermissionError):
        verified_pilot_retry(True,[],{'state':'RESOLVED_VERIFIED','repair_attempt':3})
    assert verified_pilot_retry(False,[],{})==0
