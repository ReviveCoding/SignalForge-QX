"""Budgeted CUDA diagnostics with the same physical device and resource gates."""
from contextlib import contextmanager
import json,shutil
import pandas as pd
from .runtime import admission,gpu_lease,digest,now,ensure_gpu_owner


def require_current_resources(repo,runtime,ram_bytes=512*1024**2,vram_bytes=1024**3):
    ensure_gpu_owner()
    import torch,psutil
    from .neural import require_cuda
    require_cuda()
    host=json.loads((repo/'reports/host_disk_evidence.json').read_text())
    age=pd.Timestamp.now(tz='UTC')-pd.Timestamp(host['observed_at'])
    free,total=torch.cuda.mem_get_info()
    if not pd.Timedelta(0)<=age<pd.Timedelta(hours=2) or not admission(ram_bytes,vram_bytes,64*1024**2,
        psutil.virtual_memory().available,free,total,shutil.disk_usage(runtime).free,host['free_bytes']):
        raise RuntimeError('BLOCKED_RESOURCE: current RAM/VRAM/runtime/verified host volume admission failed')
    torch.cuda.set_per_process_memory_fraction(min(.75,(free-1024**3)/total))
    return {'host_disk_evidence':host,'initial_free_vram':free,'total_vram':total,'runtime_free_disk':shutil.disk_usage(runtime).free}


@contextmanager
def research_gpu_stage(repo,runtime,stage,reservation):
    from .experiments import ComputeBudget
    profile=json.loads((repo/'configs/local_rtx4090_laptop.json').read_text())
    budget=ComputeBudget(runtime/'ledger/development_compute.sqlite',profile['budget']['development_gpu_seconds'])
    try:
        with gpu_lease(runtime):
            resources=require_current_resources(repo,runtime)
            attempt=digest({'stage':stage,'started_at':now()})
            with budget.charge(attempt,reservation):yield {'attempt_id':attempt,'resources':resources}
    finally:budget.close()
