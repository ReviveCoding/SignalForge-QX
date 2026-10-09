"""Optional train-cutoff-only next-context-feature pretraining on actual CUDA."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from .neural import Encoder,require_cuda
from .runtime import digest
from .pit import _aware_utc


def validate_pretrain(x,decisions,max_available,cutoff):
    x=np.asarray(x,dtype=np.float32);dates=pd.DatetimeIndex(_aware_utc(decisions,'SSL decisions'));known=pd.DatetimeIndex(_aware_utc(max_available,'SSL feature availability'));cut=pd.Timestamp(cutoff)
    if cut.tzinfo is None or x.ndim!=3 or x.shape[1]<2 or len(dates)!=len(x) or len(known)!=len(x) or not np.isfinite(x).all():
        raise ValueError('Valid finite pretraining sequences and temporal lineage required')
    if (dates>cut).any() or (known>dates).any() or (known>cut).any():
        raise PermissionError('SSL future disclosure/holdout cutoff violation')
    return x,digest({'x':x.tolist(),'dates':dates.astype(str).tolist(),'known':known.astype(str).tolist(),'cutoff':cut.isoformat()})


def pretrain_gru(x,decisions,max_available,cutoff,width=32,epochs=100,seed=11):
    x,data_id=validate_pretrain(x,decisions,max_available,cutoff)
    require_cuda();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Encoder(x.shape[-1],width,'gru').cuda();decoder=nn.Linear(width,x.shape[-1]).cuda()
    params=list(model.input.parameters())+list(model.sequence.parameters())+list(decoder.parameters())
    optimizer=torch.optim.AdamW(params,lr=.003);tx=torch.as_tensor(x,device='cuda');losses=[]
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        hidden,_=model.sequence(torch.nn.functional.gelu(model.input(tx[:,:-1])))
        loss=(decoder(hidden)-tx[:,1:]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite SSL loss')
        loss.backward();nn.utils.clip_grad_norm_(params,1.);optimizer.step();losses.append(float(loss.detach()))
    return {'encoder_state':{k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(('input.','sequence.'))},
            'data_id':data_id,'cutoff':str(cutoff),'method':'train_only_next_context_feature','losses':losses,
            'parameter_count':sum(p.numel() for p in params),'device':'cuda:0','synthetic_repeat_is_research_data':False}
