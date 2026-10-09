"""Validate read-only inventory against the actual qualified CUDA smoke receipt."""
from pathlib import Path


def validate_inventory(audit,stack,repo,runtime):
    if audit.get('repo')!=str(repo) or audit.get('runtime')!=str(runtime) or not audit.get('receipt_repo_matches') or not audit.get('receipt_runtime_matches'):
        raise PermissionError('Canonical repository/runtime receipt mismatch')
    if str(runtime).startswith('/mnt/') or audit.get('runtime_filesystem') not in {'ext2/ext3','ext4'} or audit.get('uid',0)<=0:
        raise PermissionError('Non-root ext4 compute runtime required')
    if not Path(audit['process_executable']).is_relative_to(Path(runtime)/'env'):
        raise PermissionError('Configured WSL environment executable required')
    if not audit.get('python','').startswith(stack['python']+'.') or not audit.get('stack_hash_matches') or not audit.get('smoke_hash_matches'):
        raise PermissionError('Resolved stack/bootstrap checksum mismatch')
    for package in ['torch','lightgbm','xgboost']:
        expected=stack[package]+('+cu126' if package=='torch' else '')
        if audit['versions'].get(package)!=expected:raise PermissionError('Implicit numerical stack change: '+package)
    smoke=audit['smoke'];checks={r['backend']:r for r in smoke.get('checks',[])}
    if smoke.get('financial_research_result') is not False or smoke.get('fixture')!='synthetic_CAPABILITY_ONLY':
        raise PermissionError('Synthetic capability receipt cannot become research evidence')
    if set(checks)!={'torch','lightgbm','xgboost'} or not all(r.get('qualified') is True and r.get('exit_code')==0 for r in checks.values()):
        raise PermissionError('All actual CUDA framework checks required')
    if any(not 0<=r.get('elapsed_seconds',-1)<=300 for r in checks.values()):raise PermissionError('Bounded capability probe receipt required')
    inventory=audit['gpu_inventory'].splitlines()
    if len(inventory)!=1 or '4090 Laptop' not in inventory[0] or '4090 Laptop' not in checks['torch'].get('device','') or checks['torch'].get('runtime_cuda')!='12.6':
        raise PermissionError('Validated single RTX 4090 Laptop CUDA identity required')
    lgb=checks['lightgbm']
    actual={(c.get('objective'),c.get('alpha')) for c in lgb.get('objective_checks',[]) if c.get('passed') is True}
    if lgb.get('device_type')!='cuda' or actual!={('regression',None),*{('quantile',q) for q in [.05,.1,.5,.9,.95]}}:
        raise PermissionError('All CUDA LightGBM objectives required; OpenCL/CPU denied')
    xgb=checks['xgboost']['objective_checks']
    if {c['objective'] for c in xgb if c.get('passed') and c.get('device')=='cuda:0'}!={'reg:squarederror','reg:quantileerror'}:
        raise PermissionError('Actual CUDA XGBoost objectives required')
    return {'state':'VERIFIED_READ_ONLY_RUNTIME_AND_CUDA_CAPABILITY','gpu_inventory':inventory[0],
        'current_GPU_availability':'REQUIRES_ACTUAL_LEASE_AND_OWNER_CHECK','physical_GPUs':1,
        'qualified_research_result':False,'actual_CUDA_capability_receipt':True}
