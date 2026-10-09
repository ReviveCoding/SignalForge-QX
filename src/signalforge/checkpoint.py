"""Optimizer-boundary checkpoints with receipt-last validation and quarantine."""
import os
from pathlib import Path
import random
import tempfile
import json
import numpy as np
import torch
from .runtime import atomic_json,file_hash,fsync_dir,now,digest


def save_checkpoint(path,state,identity):
    path=Path(path)
    if str(path.resolve()).startswith('/mnt/'):
        raise ValueError('Mutable CUDA checkpoints must reside on ext4')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,partial=tempfile.mkstemp(suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save({**state,'identity':identity,'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
                       'numpy':np.random.get_state(),'python':random.getstate()}},stream)
            stream.flush();os.fsync(stream.fileno())
        os.replace(partial,path);fsync_dir(path.parent)
        atomic_json(path.with_suffix('.receipt.json'),{'sha256':file_hash(path),'identity':identity,
                     'optimizer_step':state['step'],'sample_cursor':state['sample_cursor'],'created_at':now()})
    finally:Path(partial).unlink(missing_ok=True)


def load_checkpoint(path,identity):
    path=Path(path);receipt=path.with_suffix('.receipt.json')
    if not path.exists() and not receipt.exists():return None
    try:
        record=json.loads(receipt.read_text())
        if record['identity']!=identity or file_hash(path)!=record['sha256']:
            raise ValueError('Checkpoint identity or checksum mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['step']!=record['optimizer_step'] or state['sample_cursor']!=record['sample_cursor']:
            raise ValueError('Checkpoint boundary mismatch')
        return state
    except Exception:
        quarantine=path.parent/('quarantine-'+digest({'path':str(path),'time':now()})[:12]);quarantine.mkdir()
        for artifact in [path,receipt]:
            if artifact.exists():os.rename(artifact,quarantine/artifact.name)
        fsync_dir(quarantine)
        raise RuntimeError('Invalid checkpoint quarantined; explicit reconciliation required') from None


def restore_rng(state):
    rng=state['rng'];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    np.random.set_state(rng['numpy']);random.setstate(rng['python'])
