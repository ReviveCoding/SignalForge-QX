"""Record available project-local install/build evidence without reinstalling."""
import json,sys
import hashlib,zipfile
from pathlib import Path
from importlib.metadata import version
from signalforge.runtime import paths,file_hash,atomic_json,digest,commit_bundle,now

repo,runtime=paths()
if not Path(sys.executable).resolve().is_relative_to((runtime/'env').resolve()):raise PermissionError('Configured rootless environment required')
files={};absent=[]
for name in ['logs/bootstrap.log','logs/pip-freeze.txt','logs/conda-explicit.txt','logs/micromamba-version.txt','tools/micromamba.download.sha256']:
    path=runtime/name
    if path.is_file():files[name]={'sha256':file_hash(path),'bytes':path.stat().st_size}
    else:absent.append(name)
from lightgbm.libpath import _find_lib_path
libraries=[]
for item in _find_lib_path():
    path=Path(item).resolve()
    if not path.is_relative_to((runtime/'env').resolve()):raise PermissionError('LightGBM library outside configured environment')
    libraries.append({'relative_path':str(path.relative_to(runtime)),'sha256':file_hash(path),'bytes':path.stat().st_size})
smoke=json.loads((runtime/'logs/gpu_smoke.json').read_text())
wheel_evidence=[]
for wheel in sorted((runtime/'cache/pip/wheels').rglob('lightgbm-4.6.0-*.whl')):
    origin_path=wheel.parent/'origin.json'
    if not origin_path.is_file():continue
    origin=json.loads(origin_path.read_text())
    if not origin.get('url','').startswith('https://files.pythonhosted.org/packages/') or not origin['url'].endswith('/lightgbm-4.6.0.tar.gz'):
        raise ValueError('Unexpected retained LightGBM source origin')
    source_hash=origin['archive_info']['hashes']['sha256']
    if len(source_hash)!=64 or any(c not in '0123456789abcdef' for c in source_hash):raise ValueError('Invalid source hash')
    with zipfile.ZipFile(wheel) as archive:
        names=[n for n in archive.namelist() if n.endswith('/lib_lightgbm.so')]
        if len(names)!=1:raise ValueError('Unique cached CUDA library required')
        library_hash=hashlib.sha256(archive.read(names[0])).hexdigest()
    if library_hash not in {item['sha256'] for item in libraries}:raise ValueError('Cached and installed LightGBM library mismatch')
    wheel_evidence.append({'wheel_relative_path':str(wheel.relative_to(runtime)),'wheel_sha256':file_hash(wheel),
        'origin_metadata_sha256':file_hash(origin_path),'source_url':origin['url'],'source_archive_sha256':source_hash,
        'library_sha256':library_hash,'installed_library_exact_match':True})
checks={r['backend']:r for r in smoke['checks']}
if not checks['lightgbm']['qualified'] or checks['lightgbm']['device_type']!='cuda':raise PermissionError('Actual CUDA source-build capability evidence missing')
result={'state':'RECORDED_EXISTING_ROOTLESS_INSTALL_EVIDENCE','python_executable':str(Path(sys.executable).resolve()),
    'resolved_versions':{p:version(p) for p in ['torch','lightgbm','xgboost','numpy','pandas']},
    'install_records':files,'missing_records':absent,'lightgbm_libraries':libraries,
    'current_bootstrap_script_sha256':file_hash(repo/'scripts/bootstrap_wsl.sh'),
    'stack_spec_sha256':file_hash(repo/'configs/stack_candidate.json'),'CUDA_capability_receipt_sha256':file_hash(runtime/'logs/gpu_smoke.json'),
    'registered_source_build_command':'python -m pip install lightgbm==4.6.0 --no-binary=lightgbm --no-build-isolation --config-settings=cmake.define.USE_CUDA=ON --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89',
    'retained_source_build_wheels':wheel_evidence,
    'source_archive_original_hash_retained':bool(wheel_evidence),'historical_bootstrap_script_hash_retained':False,
    'claim_boundary':'Actual rootless install records, pip-retained source origin hash, and cached CUDA library matched byte-for-byte to the installed library. The historical executed bootstrap-script digest remains unavailable.',
    'new_installs':0,'global_driver_or_configuration_mutations':0,'CUDA_allocations_by_this_audit':0,'created_at':now()}
identity=digest(result);commit_bundle(runtime/'artifacts/rootless_install_audits'/identity,{'audit.json':result},{'runtime':str(runtime)})
result['artifact_id']=identity;atomic_json(repo/'reports/rootless_install_evidence.json',result)
print(json.dumps(result,indent=2))
