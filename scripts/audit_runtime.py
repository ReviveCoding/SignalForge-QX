"""Read-only runtime reconciliation; never exposes credential values."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

root = Path(os.environ['SIGNALFORGE_REPO'])
runtime = Path(os.environ['SIGNALFORGE_RUNTIME'])
receipt = json.loads((root / '.local/bootstrap_receipt.json').read_text(encoding='utf-8-sig'))
smoke = runtime / 'logs/gpu_smoke.json'
print(json.dumps({
    'repo': str(root), 'runtime': str(runtime), 'python': platform.python_version(),
    'uid': os.getuid(), 'runtime_filesystem': subprocess.check_output(['stat', '-f', '-c', '%T', str(runtime)], text=True).strip(),
    'runtime_disk': shutil.disk_usage(runtime)._asdict(),
    'receipt_repo_matches': receipt['code_repo'] == str(root),
    'receipt_runtime_matches': receipt['runtime'] == str(runtime),
    'stack_hash_matches': hashlib.sha256((root / 'configs/stack_candidate.json').read_bytes()).hexdigest() == receipt['stack_spec_sha256'],
    'smoke_hash_matches': hashlib.sha256(smoke.read_bytes()).hexdigest() == receipt['gpu_smoke_sha256'],
    'smoke': json.loads(smoke.read_text()),
    'versions': {p: md.version(p) for p in ['torch','lightgbm','xgboost','numpy','pandas','scipy','pytest','requests']},
    'resolved_packages': {d.metadata['Name']:d.version for d in md.distributions()},
    'dispatcher_sha256': hashlib.sha256((root/'scripts/dispatch_wsl.sh').read_bytes()).hexdigest(),
    'process_executable': sys.executable,
    'sqlite_import_qualified': __import__('sqlite3').sqlite_version,
    'credential_presence': {k: bool(os.environ.get(k)) for k in ['FRED_API_KEY','TIINGO_API_TOKEN','ALPHAVANTAGE_API_KEY','SEC_USER_AGENT']},
    'gpu_inventory': subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.free,driver_version','--format=csv,noheader'], text=True).strip(),
    'gpu_processes': subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'], text=True).strip(),
}, indent=2))
