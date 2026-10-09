"""CUDA optimizer-boundary worker for controlled project-owned process interruption."""
import argparse
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from signalforge.neural import Encoder,normalized_loss,require_cuda
from signalforge.runtime import file_hash,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--steps',type=int,default=16);p.add_argument('--pause-step',type=int,default=0);a=p.parse_args()
    require_cuda();torch.set_num_threads(4)
    torch.manual_seed(731);torch.cuda.manual_seed_all(731);np.random.seed(731);random.seed(731)
    model=Encoder(4,16,'mlp').cuda();opt=torch.optim.AdamW(model.parameters(),lr=.002)
    x=torch.randn(32,4,device='cuda');y=.3*x[:,0]
    path=Path(a.checkpoint);step=0
    if path.exists():
        receipt=path.with_suffix('.receipt.json')
        import json
        if file_hash(path)!=json.loads(receipt.read_text())['sha256']:
            raise ValueError('Corrupt checkpoint; preserve for quarantine')
        checkpoint=torch.load(path,weights_only=False)
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        step=checkpoint['step']
        if checkpoint['sample_cursor']!=step*len(y):raise ValueError('Sample cursor mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    while step<a.steps:
        # RNG participates in training, so recovery tests actual RNG restoration.
        noisy=x+.01*torch.randn_like(x)
        opt.zero_grad();loss=normalized_loss(y,*model(noisy),1);loss.backward();opt.step();step+=1
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'sample_cursor':step*len(y),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                        'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},f)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        atomic_json(path.with_suffix('.receipt.json'),{'sha256':file_hash(path),'step':step,'sample_cursor':step*len(y)})
        if step==a.pause_step:
            print('SAFE_BOUNDARY '+str(step),flush=True)
            time.sleep(60)
    print('COMPLETE '+str(step),flush=True)


if __name__=='__main__':main()
