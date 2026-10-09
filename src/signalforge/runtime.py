"""Durable artifacts, runtime paths, admission and a cross-framework CUDA lease."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # Windows readers on /mnt/c may briefly deny delete sharing. Retry
        # the same flushed file; never publish in place or swallow a denial.
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.05 * 2**min(attempt, 3))
        fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def fsync_dir(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def paths():
    repo = Path(os.environ['SIGNALFORGE_REPO']).resolve()
    runtime = Path(os.environ['SIGNALFORGE_RUNTIME']).resolve()
    if str(runtime).startswith('/mnt/') or not runtime.is_relative_to(Path.home()):
        raise ValueError('Heavy state must reside under Linux HOME on ext4')
    return repo, runtime


def code_hash(repo):
    files = sorted(list((repo / 'src').rglob('*.py')) + list((repo / 'scripts').glob('*.py')))
    return digest({str(p.relative_to(repo)): file_hash(p) for p in files})


def admission(ram, vram, disk, available_ram, free_vram, total_vram, free_disk, host_free):
    gib = 1024**3
    return (ram <= min(8*gib, .6*available_ram) and
            vram <= max(0, min(12*gib, total_vram-2*gib, free_vram-gib)) and
            disk <= free_disk-10*gib and host_free is not None and disk <= host_free-20*gib)


def ensure_gpu_owner():
    inventory=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,timeout=10)
    lines=[p.strip() for p in inventory.stdout.splitlines() if p.strip()]
    malformed=any(not p.isdigit() for p in lines)
    pids={int(p) for p in lines if p.isdigit()}
    if inventory.returncode or malformed or pids-{os.getpid()}:
        raise RuntimeError('BLOCKED_GPU: unrelated CUDA owner or unknown inventory')


def check_shutdown_history(runtime,previous,boot_id):
    """Retain unexplained interrupted boot changes; never restart heavy work forever."""
    path=Path(runtime)/'ledger/gpu_shutdown_history.json'
    history=json.loads(path.read_text()) if path.exists() else {'incidents':[],'consecutive_unexplained':0,'state':'READY'}
    if previous.get('state')=='RUNNING' and previous.get('boot_id')!=boot_id:
        identity=digest(previous)
        if identity not in {i['identity'] for i in history['incidents']}:
            history['incidents'].append({'identity':identity,'previous_lease':previous,'observed_boot_id':boot_id,'observed_at':now(),
                                        'cause':'Unexplained WSL/host restart; host shutdown not independently attributed'})
            history['consecutive_unexplained']+=1
        if history['consecutive_unexplained']>=2:history['state']='PAUSED_RECOVERY'
        atomic_json(path,history)
    if history['state']=='PAUSED_RECOVERY':
        raise RuntimeError('PAUSED_RECOVERY: repeated unexplained boot interruptions; diagnose before heavy restart')
    return history


@contextmanager
def gpu_lease(runtime):
    import fcntl
    lock_path = Path(runtime) / 'locks/gpu0.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('BLOCKED_GPU: CUDA lease held') from None
        started=False
        try:
            ensure_gpu_owner()
            lock.seek(0);content=lock.read()
            previous=json.loads(content) if content else {}
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            check_shutdown_history(runtime,previous,boot_id)
            record={'pid':os.getpid(),'boot_id':boot_id,'started_at':now(),'state':'RUNNING'}
            lock.seek(0)
            lock.truncate()
            lock.write(json.dumps(record));lock.flush();os.fsync(lock.fileno());started=True
            yield
        finally:
            if started:
                record.update(state='RELEASED',ended_at=now())
                lock.seek(0);lock.truncate();lock.write(json.dumps(record));lock.flush();os.fsync(lock.fileno())
                path=Path(runtime)/'ledger/gpu_shutdown_history.json'
                if path.exists():
                    history=json.loads(path.read_text());history['consecutive_unexplained']=0;atomic_json(path,history)
            fcntl.flock(lock, fcntl.LOCK_UN)


def commit_bundle(directory, files, metadata):
    """Receipt last; readers ignore bundles lacking a valid final manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'receipt.json'
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        validate_bundle(directory)
        if existing.get('metadata') != metadata:
            raise ValueError('Immutable bundle identity conflict')
        return existing
    partials=list(directory.glob('*.partial'))
    existing_files=[directory/name for name in files if (directory/name).exists()]
    if partials or existing_files:
        quarantine=directory/('quarantine-'+digest({'time':now(),'pid':os.getpid()})[:12])
        quarantine.mkdir()
        for orphan in partials+existing_files:
            os.rename(orphan,quarantine/orphan.name)
        fsync_dir(quarantine)
    for name, value in files.items():
        if Path(name).name != name or name == 'receipt.json':
            raise ValueError('Invalid artifact name')
        if isinstance(value, bytes):
            atomic_bytes(directory / name, value)
        else:
            atomic_json(directory / name, value)
    result = {'metadata': metadata, 'artifacts': {n: file_hash(directory / n) for n in files}}
    atomic_json(receipt, result)
    return result


def validate_bundle(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'receipt.json').read_text())
    if not receipt['artifacts']:
        raise ValueError('Empty artifact manifest')
    for name, expected in receipt['artifacts'].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError('Artifact checksum mismatch')
    return receipt
