"""Build a private 4.6.0 CUDA backend with one pinned memory-safety correction."""
import json,os,subprocess,sys,tarfile,zipfile,argparse,shutil
from pathlib import Path
from signalforge.runtime import paths,file_hash,atomic_json,commit_bundle,digest,now

repo,runtime=paths();parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,choices=[1,2],default=1);parser.add_argument('--manual-boundary-review',action='store_true');args=parser.parse_args()
base=runtime/'tools/successor_lightgbm_repair';root=base if args.attempt==1 else runtime/'tools/successor_lightgbm_repair_attempt2'
if args.manual_boundary_review:
    authorization=repo/'.local/ALLOW_MANUAL_CUDA_BOUNDARY_REPAIR'
    if not authorization.exists():raise PermissionError('AUTH_GATE: additional build requires explicit manual authorization beyond automatic repair ceiling')
    grant=json.loads(authorization.read_text())
    if grant.get('authorized') is not True or grant.get('scope')!='one_manual_cuda_boundary_repair_build' or grant.get('automatic_repair_ceiling')!=2 or grant.get('reserved_access') is not False or grant.get('gpu_budget_increase') is not False:
        raise PermissionError('AUTH_GATE: manual boundary review scope invalid')
    root=runtime/'tools/manual_cuda_boundary_repair'
if args.attempt==2:
    root.mkdir(exist_ok=True)
    for name in ['source_receipt.json','lightgbm-4.6.0.tar.gz','upstream.patch']:
        destination=root/name
        if not destination.exists():shutil.copyfile(base/name,destination)
origin=json.loads((root/'source_receipt.json').read_text())
archive=root/'lightgbm-4.6.0.tar.gz';patch=root/'upstream.patch'
if file_hash(archive)!=origin['source_sha256'] or file_hash(patch)!=origin['patch_sha256']:
    raise PermissionError('INTEGRITY_OR_HASH_GATE: repair source changed')
source=root/'source';source.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
checkout=source/'lightgbm-4.6.0';target=checkout/'include/LightGBM/cuda/cuda_algorithms.hpp'
old='exclusive_result = shared_mem_buffer[warpLane - 1];'
new='exclusive_result = shared_mem_buffer[warpID - 1];'
text=target.read_text();before=file_hash(target)
if text.count(old)!=1:raise PermissionError('Unexpected original CUDA prefix-sum source')
if ('-    '+old) not in patch.read_text() or ('+    '+new) not in patch.read_text():
    raise PermissionError('Pinned upstream patch differs from single-index correction')
target.write_text(text.replace(old,new));after=file_hash(target)
corrections=[]
if args.attempt==2:
    original=target.read_text()
    before_prefix='''  const REDUCE_VAL_T thread_base = shared_buffer[threadIdx.x];
  for (INDEX_T index = start; index < end; ++index) {
    out_values[index] = thread_base + static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }'''
    after_prefix='''  REDUCE_VAL_T running_sum = thread_sum;
  for (INDEX_T index = start; index < end; ++index) {
    running_sum += static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
    out_values[index] = running_sum;
  }'''
    if original.count(before_prefix)!=1:raise PermissionError('Exact sorted-prefix correction source mismatch')
    target.write_text(original.replace(before_prefix,after_prefix))
    corrections.append({'path':str(target.relative_to(checkout)),'before_sha256':after,'after_sha256':file_hash(target),
                        'reason':'32-slot per-warp shared buffer cannot be indexed by 512 thread IDs; use returned exclusive per-thread sum and accumulate each sorted weight'})
    if args.manual_boundary_review:
        before_boundary=file_hash(target);original=target.read_text()
        old_boundary='''    if (pos == 0 || pos == len - 1) {
      *out_value = values[pos];
    }'''
        new_boundary='''    if (pos == 0) {
      *out_value = values[sorted_indices[0]];
      return;
    }'''
        if original.count(old_boundary)!=1:raise PermissionError('Exact global weighted boundary source mismatch')
        target.write_text(original.replace(old_boundary,new_boundary))
        corrections.append({'path':str(target.relative_to(checkout)),'before_sha256':before_boundary,'after_sha256':file_hash(target),'reason':'First sorted CDF position must return before accessing index -1; preserve valid interior interpolation'})
        device=checkout/'src/cuda/cuda_algorithms.cu';original=device.read_text();before_device=file_hash(device)
        old_device='''    if (pos == 0 || pos == len - 1) {
      return values[pos];
    }'''
        new_device='''    if (pos == 0 || pos == len - 1) {
      return values[indices[pos]];
    }'''
        if original.count(old_device)!=1:raise PermissionError('Exact sorted leaf boundary source mismatch')
        device.write_text(original.replace(old_device,new_device))
        corrections.append({'path':str(device.relative_to(checkout)),'before_sha256':before_device,'after_sha256':file_hash(device),'reason':'Leaf boundary reads sorted support, not an arbitrary original row'})
    cmake=checkout/'CMakeLists.txt';original_cmake=cmake.read_text();before_cmake=file_hash(cmake)
    anchor='    message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHS}")'
    if original_cmake.count(anchor)!=1:raise PermissionError('CUDA build target source mismatch')
    cmake.write_text(original_cmake.replace(anchor,'    set(CUDA_ARCHS "89-real")\n'+anchor))
    corrections.append({'path':'CMakeLists.txt','before_sha256':before_cmake,'after_sha256':file_hash(cmake),
                        'reason':'Private successor build targets only the actual SM89 GPU; original environment untouched'})
wheel_dir=root/'wheels';wheel_dir.mkdir(exist_ok=True)
environment={**os.environ,'CMAKE_BUILD_PARALLEL_LEVEL':'2'}
command=[sys.executable,'-m','pip','wheel',str(checkout),'--no-deps','--no-build-isolation',
         '--config-settings=cmake.define.USE_CUDA=ON',
         '--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89',
         '--config-settings=build.tool-args=-j2','--wheel-dir',str(wheel_dir)]
with (root/'build.log').open('w') as log:
    result=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
if result.returncode:raise RuntimeError('Private CUDA build failed; retained log: '+str(root/'build.log'))
wheels=list(wheel_dir.glob('lightgbm-4.6.0-*.whl'))
if len(wheels)!=1:raise RuntimeError('Unique private CUDA wheel required')
wheel=wheels[0];package=root/'python';package.mkdir(exist_ok=True)
with zipfile.ZipFile(wheel) as z:
    for name in z.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()):raise PermissionError('Wheel path escape')
    z.extractall(package)
library=package/'lightgbm/lib/lib_lightgbm.so'
report={'state':'BUILT_PRIVATE_SUCCESSOR_CUDA_BACKEND','created_at':now(),'version':'4.6.0',
        'source':origin,'original_header_sha256':before,'patched_header_sha256':after,
        'repair_attempt':args.attempt,'additional_corrections':corrections,
        'repair_mode':'MANUALLY_AUTHORIZED_BOUNDARY_REPAIR' if args.manual_boundary_review else 'BOUNDED_AUTOMATIC_REPAIR',
        'patch_scope':'ShufflePrefixSumExclusive first lane: previous warp index instead of negative lane index',
        'native_library_sha256':file_hash(library),'wheel_sha256':file_hash(wheel),
        'package_relative_path':str(package.relative_to(runtime)),
        'package_hashes':{str(p.relative_to(package)):file_hash(p) for p in sorted(package.rglob('*')) if p.is_file()},
        'device_type':'cuda','cuda_architecture':89,'build_parallel_jobs':2,
        'original_environment_modified':False,'weights_or_regularization_changed':False,
        'reserved_access':False,'qualified_for_final':False}
identity=digest(report)
receipt=commit_bundle(runtime/'artifacts/successor_cuda_backends'/identity,
    {'backend.json':report,'upstream.patch':patch.read_bytes(),'build.log':(root/'build.log').read_bytes()},
    {'backend_id':identity,'scope':'successor_only'})
atomic_json(repo/'reports/successor_cuda_backend.json',{**report,'backend_id':identity,'receipt_id':digest(receipt)})
print(json.dumps({'state':report['state'],'backend_id':identity,'native_library_sha256':report['native_library_sha256']}))
