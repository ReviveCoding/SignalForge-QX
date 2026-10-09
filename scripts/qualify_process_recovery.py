"""Kill only the child created here, resume the same optimizer/RNG boundary, compare."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch
from signalforge.runtime import paths,gpu_lease,atomic_json,file_hash,now


def main():
    repo,runtime=paths();folder=Path(tempfile.mkdtemp(prefix='recovery-',dir=runtime/'artifacts'))
    with gpu_lease(runtime):
        command=[sys.executable,'scripts/recovery_worker.py']
        interrupted=folder/'interrupted.pt';reference=folder/'reference.pt'
        proc=subprocess.Popen(command+['--checkpoint',str(interrupted),'--pause-step','8'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            import selectors
            sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ)
            if not sel.select(timeout=30):raise RuntimeError('Boundary signal timeout')
            line=proc.stdout.readline().strip()
            if line!='SAFE_BOUNDARY 8':raise RuntimeError('Unexpected worker boundary: '+line)
            proc.terminate();proc.wait(timeout=10)
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=10)
        boundary=json.loads(interrupted.with_suffix('.receipt.json').read_text())
        for path in [reference,interrupted]:
            subprocess.run(command+['--checkpoint',str(path)],check=True,timeout=60,capture_output=True,text=True)
        a=torch.load(reference,map_location='cpu',weights_only=False)
        b=torch.load(interrupted,map_location='cpu',weights_only=False)
        delta=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
        report={'created_at':now(),'evidence_kind':'synthetic_fixture','device':'cuda:0',
                'interrupted_at_step':boundary['step'],'restored_sample_cursor':boundary['sample_cursor'],
                'final_steps':[a['step'],b['step']],'max_parameter_abs_delta':delta,
                'passed':delta==0 and a['step']==b['step']==16,
                'reference_path':str(reference),'reference_sha256':file_hash(reference),
                'resumed_path':str(interrupted),'resumed_sha256':file_hash(interrupted),
                'scope':'actual project-owned CUDA process termination and same-stack resume; not hardware shutdown recovery'}
        atomic_json(repo/'reports/process_recovery_qualification.json',report)
        print(json.dumps(report,indent=2))
        return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
