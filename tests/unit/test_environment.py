import copy,json
from pathlib import Path
import pytest
from signalforge.environment import validate_inventory


def fixture():
    repo=Path(__file__).parents[2]
    audit=json.loads((repo/'reports/environment_audit.json').read_text());stack=json.loads((repo/'configs/stack_candidate.json').read_text())
    return audit,stack,Path(audit['repo']),Path(audit['runtime'])


def test_inventory_wrong_gpu_cpu_opencl_or_objective_denied():
    audit,stack,repo,runtime=fixture();result=validate_inventory(audit,stack,repo,runtime)
    assert result['physical_GPUs']==1 and 'GPU_currently_free' not in result
    for field,value in [('gpu_inventory','NVIDIA GeForce RTX 3090'),('uid',0),('runtime_filesystem','9p')]:
        changed=copy.deepcopy(audit);changed[field]=value
        with pytest.raises(PermissionError):validate_inventory(changed,stack,repo,runtime)
    changed=copy.deepcopy(audit);changed['smoke']['checks'][1]['device_type']='gpu'
    with pytest.raises(PermissionError,match='OpenCL'):validate_inventory(changed,stack,repo,runtime)
    changed=copy.deepcopy(audit);changed['smoke']['checks'][1]['objective_checks'].pop()
    with pytest.raises(PermissionError,match='objectives'):validate_inventory(changed,stack,repo,runtime)
    changed=copy.deepcopy(audit);changed['repo']='/other/editable/clone'
    with pytest.raises(PermissionError,match='repository/runtime'):validate_inventory(changed,stack,repo,runtime)
    changed=copy.deepcopy(audit);changed['process_executable']='/usr/bin/python3'
    with pytest.raises(PermissionError,match='environment executable'):validate_inventory(changed,stack,repo,runtime)


def test_inventory_environment_drift_and_fixture_research_claim_denied():
    audit,stack,repo,runtime=fixture()
    for field,value in [('torch','2.7.1+cpu'),('lightgbm','9.0.0')]:
        changed=copy.deepcopy(audit);changed['versions'][field]=value
        with pytest.raises(PermissionError,match='stack change'):validate_inventory(changed,stack,repo,runtime)
    changed=copy.deepcopy(audit);changed['smoke']['financial_research_result']=True
    with pytest.raises(PermissionError,match='research evidence'):validate_inventory(changed,stack,repo,runtime)
    changed=copy.deepcopy(audit);changed['smoke']['checks'][0]['elapsed_seconds']=301
    with pytest.raises(PermissionError,match='Bounded'):validate_inventory(changed,stack,repo,runtime)
