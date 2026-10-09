"""Hash-verified private CUDA library activation, restricted to successor workers."""
import json
import sys
from pathlib import Path
from .runtime import file_hash, validate_bundle, digest


def verify_backend(repo, runtime):
    pointer = Path(repo)/'reports/successor_cuda_backend.json'
    if not pointer.exists():
        return None
    report = json.loads(pointer.read_text())
    root = Path(runtime).resolve()
    package = (root/report['package_relative_path']).resolve()
    if not package.is_relative_to(root) or report.get('version') != '4.6.0':
        raise PermissionError('INTEGRITY_OR_HASH_GATE: private backend path/version')
    bundle = root/'artifacts/successor_cuda_backends'/report['backend_id']
    receipt = validate_bundle(bundle)
    if digest(receipt) != report['receipt_id']:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: private backend receipt')
    frozen = json.loads((bundle/'backend.json').read_text())
    if frozen != {k:v for k,v in report.items() if k not in {'backend_id','receipt_id'}}:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: private backend pointer')
    for relative, expected in report['package_hashes'].items():
        path = (package/relative).resolve()
        if not path.is_relative_to(package) or file_hash(path) != expected:
            raise PermissionError('INTEGRITY_OR_HASH_GATE: private backend file')
    return report


def activate_backend(repo, runtime):
    report=verify_backend(repo,runtime)
    if report is None:return None
    package=(Path(runtime)/report['package_relative_path']).resolve()
    loaded = sys.modules.get('lightgbm')
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(package):
        raise PermissionError('INTEGRITY_OR_HASH_GATE: original backend already imported')
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    return report


def operator_id(repo, study, track, family, backend):
    from .track_engine import track_operator_id
    registered = track_operator_id(repo, study, track)
    return digest({'registered_operator':registered,'native_backend_id':backend['backend_id']}) if backend and family == 'lightgbm' else registered
