import json
from pathlib import Path
from signalforge.runtime import file_hash


def test_native_node_checksum_mismatch_and_windows_archive_escape_denied():
    repo=Path(__file__).resolve().parents[2]
    # WSL PE interop is unavailable here. Execute the Windows fixture via the
    # native PowerShell host, then verify its actual current-source receipt.
    receipt=repo/'reports/native_archive_test_execution.json'
    assert receipt.is_file(),'Run tests/native/Test-ArchiveSafety.ps1 using native PowerShell 7 first'
    result=json.loads(receipt.read_text())
    assert result['state']=='PASSED_SYNTHETIC_NATIVE_ARCHIVE_FIXTURES'
    assert result['native_windows'] and int(result['powershell'].split('.')[0])>=7
    assert result['case_count']==14 and not result['actual_Node_distribution_downloaded']
    for name,sha in result['file_hashes'].items():assert file_hash(repo/name)==sha
