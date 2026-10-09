from pathlib import Path

ROOT=Path(__file__).parents[2]


def test_progress_monitor_reports_worker_instead_of_micromamba_launcher():
    import subprocess
    monitor=(ROOT/'scripts/Watch-SignalForgeProgress.ps1').read_text()
    body=monitor.split('function Get-WslLine',1)[1].split('function Parse-Proc',1)[0]
    assert "grep -v '[m]icromamba run' | head -1" in body
    processes=('628399 628398 02:04 0.0 0.0 13356 Ssl+ /tools/bin/micromamba run -p /env python scripts/run_successor_gpu_pilot.py\n'
               '628401 628399 02:04 102 0.1 304120 Rl+ python scripts/run_successor_gpu_pilot.py --track Main-A\n')
    selected=subprocess.run(['bash','-c',"grep '[r]un_successor_gpu_pilot.py' | grep -v '[m]icromamba run' | head -1"],
                            input=processes,capture_output=True,text=True,check=True).stdout
    assert selected.split()[0]=='628401'
    assert selected.split()[3]=='102'


def test_recovery_watcher_preserves_scientific_and_gpu_boundaries():
    watcher=(ROOT/'scripts/Watch-SignalForgeRecovery.ps1').read_text()
    prompt=(ROOT/'scripts/SignalForge-Recovery-Prompt.txt').read_text()

    assert '[int]$MaxRepairAttempts=2' in watcher
    assert "PAUSE_BEFORE_GPU" in watcher
    assert "track-development','--track','both','--input-version','v311" in watcher
    assert "run_successor_gpu_pilot.py" not in watcher
    assert "development_gpu_seconds" not in watcher
    assert "budget ceiling/reset/refund mutation" in watcher

    assert "Never delete, refund, reset, rewrite, or enlarge any existing compute ledger" in prompt
    assert "Never weaken source, hash, PIT, clock, maturity, reserved-data, freeze, validation, qualification, GPU lease, or final-admission gates" in prompt
    assert "Do not dispatch training from this repair agent" in prompt
    assert "ALLOW_SUCCESSOR_GPU" in prompt
    assert '21600-second' in prompt
    assert '"state": "FIX_VERIFIED" | "SAFE_RETRY" | "MANUAL_BLOCKED"' in prompt
    assert "Do NOT start the long-running Main/Nested run yourself." in prompt


def test_host_disk_evidence_is_written_without_utf8_bom():
    script=(ROOT/'scripts/Write-HostDiskEvidence.ps1').read_text()
    assert 'New-Object Text.UTF8Encoding($false)' in script
    assert 'Set-Content -LiteralPath $temporary -Encoding utf8' not in script
