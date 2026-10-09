import json
import os
import pytest
from signalforge.cli import main
from signalforge.reporting import render,validate_lineage
from signalforge.runtime import atomic_json


def setup_repo(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setenv('SIGNALFORGE_REPO',str(repo));monkeypatch.setenv('SIGNALFORGE_RUNTIME',str(runtime))
    # Production paths require HOME; point explicit test HOME to the synthetic root.
    monkeypatch.setenv('HOME',str(tmp_path))
    return repo,runtime


def test_incremental_report_blockers_and_lineage(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    atomic_json(repo/'reports/public_pilot.json',{'sources':[{'source':'fred','state':'BLOCKED_AUTH','reason':'key absent'}]})
    render(repo);assert validate_lineage(repo)
    text=(repo/'reports/technical_report.md').read_text()
    assert 'BLOCKED_AUTH' in text and 'INCOMPLETE' in text and 'BLOCKED_PRICE_P1' in text
    atomic_json(repo/'reports/public_pilot.json',{'sources':[]})
    assert validate_lineage(repo)
    with pytest.raises(ValueError):validate_lineage(repo,require_current=True)


def test_cli_freeze_and_forward_fail_closed(tmp_path,monkeypatch):
    setup_repo(tmp_path,monkeypatch)
    assert main(['freeze','--study','sgqx-v3'])==2
    assert main(['forward-once','--as-of','NOW'])==2
    assert main(['pipeline','--mode','locked','--authorize-final-read'])==2


def test_cli_readonly_and_report(tmp_path,monkeypatch):
    repo,runtime=setup_repo(tmp_path,monkeypatch)
    assert main(['reconcile','--read-only'])==0
    assert not (runtime/'ledger/final_access.json').exists()
    assert main(['report','--study','sgqx-v3','--validate-lineage'])==0


def test_report_completion_dimensions_and_status_snapshot_lineage(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    status={'implementation_complete':False,'real_data_acquisition_qualified':False,'strict_PIT_available':False,
        'single_gpu_qualified_on_user_device':True,'reserved_evaluation_complete':False,'freeze_state':'BLOCKED_FREEZE_GATES'}
    atomic_json(repo/'IMPLEMENTATION_STATUS.json',status)
    render(repo);assert validate_lineage(repo,require_current=True)
    text=(repo/'reports/technical_report.md').read_text()
    assert '| Actual local CUDA capability | True |' in text and '| Reserved batch completed | False |' in text
    assert 'NOT_OBSERVED' in text and 'NOT_EXECUTED' in text and 'BLOCKED_PRICE_P1' in text
    atomic_json(repo/'IMPLEMENTATION_STATUS.json',{**status,'implementation_complete':True})
    assert validate_lineage(repo)
    with pytest.raises(ValueError):validate_lineage(repo,require_current=True)

def test_nport_report_matches_current_conservative_clock(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    atomic_json(repo/'reports/fred_segmented_plan.json',{'state':'REGISTERED','segment_count':20,'series_count':5})
    atomic_json(repo/'reports/nport_bulk_integration_v311.json',{'state':'SUCCEEDED_RECONSTRUCTED_NPORT_TARGET_FLOWS','event_rows':1188,
              'clock_policy':'edgar_next_federal_business_day_235959_et_development_v311'})
    render(repo);text=(repo/'reports/technical_report.md').read_text()
    assert 'next U.S. federal business day' in text
    assert 'EDGAR Accepted +24h' not in text
