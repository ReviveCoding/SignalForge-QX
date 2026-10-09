"""A Windows reader must not turn a completed fit into a lost receipt.

The actual WSL/NTFS EACCES reproduction is retained separately in .local.
These fixtures verify bounded retry and fail-closed publication semantics.
"""
import errno
from pathlib import Path
from unittest.mock import patch

import pytest

from signalforge.runtime import atomic_bytes


def test_transient_reader_lock_retries_same_pending_payload(tmp_path):
    target = tmp_path / 'receipt.json'
    target.write_bytes(b'old')
    import os
    real_replace = os.replace
    attempts = []

    def replace(source, destination):
        attempts.append((str(source), Path(source).read_bytes()))
        assert target.read_bytes() == b'old'
        if len(attempts) < 3:
            raise PermissionError(errno.EACCES, 'Windows reader holds delete lock')
        return real_replace(source, destination)

    with patch('signalforge.runtime.os.replace', side_effect=replace), patch('time.sleep') as sleep:
        atomic_bytes(target, b'complete new receipt')
    assert target.read_bytes() == b'complete new receipt'
    assert len(attempts) == 3
    assert len({source for source, _ in attempts}) == 1
    assert all(payload == b'complete new receipt' for _, payload in attempts)
    assert sleep.call_count == 2
    assert not list(tmp_path.glob('*.partial'))


def test_persistent_permission_error_is_bounded_and_preserves_old_receipt(tmp_path):
    target = tmp_path / 'receipt.json'
    target.write_bytes(b'old')
    with patch('signalforge.runtime.os.replace', side_effect=PermissionError(errno.EACCES, 'persistent denial')) as replace, patch('time.sleep') as sleep:
        with pytest.raises(PermissionError):
            atomic_bytes(target, b'new')
    assert replace.call_count == 8
    assert sleep.call_count == 7
    assert sum(call.args[0] for call in sleep.call_args_list) < 2.1
    assert target.read_bytes() == b'old'
    assert not list(tmp_path.glob('*.partial'))


def test_structural_replace_error_is_not_retried(tmp_path):
    target = tmp_path / 'receipt.json'
    target.write_bytes(b'old')
    with patch('signalforge.runtime.os.replace', side_effect=OSError(errno.EINVAL, 'invalid operation')) as replace, patch('time.sleep') as sleep:
        with pytest.raises(OSError):
            atomic_bytes(target, b'new')
    assert replace.call_count == 1
    sleep.assert_not_called()
    assert target.read_bytes() == b'old'


def test_successful_publication_does_not_wait(tmp_path):
    target = tmp_path / 'receipt.json'
    with patch('time.sleep') as sleep:
        atomic_bytes(target, b'new')
    assert target.read_bytes() == b'new'
    sleep.assert_not_called()


def test_full_supervisor_failure_progress_uses_full_ledger(tmp_path):
    from signalforge.successor_runtime import progress
    bill = {'charged_seconds': 3211., 'ceiling_seconds': 21600., 'charges': []}
    with patch('signalforge.successor_runtime.ledger_snapshot', return_value=bill) as snapshot, patch('signalforge.successor_runtime.gpu_context', return_value={}):
        result = progress(tmp_path, tmp_path, 'SUCCESSOR GPU FULL', status='BLOCKED_PRESERVED_WORKER_FAILURE')
    assert snapshot.call_args.args == (tmp_path / 'ledger/successor_gpu_full_compute.sqlite', 21600)
    assert result['remaining_gpu_seconds'] == 18389.
