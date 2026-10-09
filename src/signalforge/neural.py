"""Compact GRU-D/TFT-style and RGMF implementations, not paper reproductions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def ordered_head(raw):
    m=raw[...,1]
    a,b,c,d=F.softplus(raw[...,2:]).unbind(-1)
    return raw[...,0],torch.stack([m-a-b,m-a,m,m+c,m+c+d],dim=-1)


class Encoder(nn.Module):
    def __init__(self,features,width=32,kind='mlp'):
        super().__init__()
        self.kind=kind
        self.input=nn.Linear(features,width)
        if kind in {'gru','gru_d','tft'}:
            self.sequence=nn.GRU(width,width,batch_first=True)
        if kind in {'transformer','tft'}:
            self.attention=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True),1)
        if kind=='gru_d':
            self.decay=nn.Parameter(torch.zeros(features))
        if kind=='tft':
            self.variable_gate=nn.Linear(features,features)
            self.output_gate=nn.Linear(width,width)
        self.output=nn.Linear(width,6)

    def forward(self,x,observed=None,elapsed=None,padding=None):
        if x.ndim==2:
            x=x[:,None,:]
        if self.kind=='gru_d':
            if observed is None or elapsed is None:
                raise ValueError('GRU-D-style requires observed mask and elapsed times')
            last=torch.zeros_like(x[:,0])
            values=[]
            for t in range(x.shape[1]):
                dec=torch.exp(-F.relu(self.decay)*elapsed[:,t])
                last=torch.where(observed[:,t],x[:,t],last*dec)
                values.append(last)
            x=torch.stack(values,dim=1)
        if self.kind=='tft':
            x=x*torch.softmax(self.variable_gate(x),dim=-1)*x.shape[-1]
        z=self.input(x)
        if self.kind!='linear':
            z=F.gelu(z)
        if self.kind in {'gru','gru_d','tft'}:
            z,_=self.sequence(z)
        if self.kind in {'transformer','tft'}:
            positions=torch.arange(z.shape[1],device=z.device,dtype=z.dtype)
            z=z+torch.sin(positions[:,None]/(10000**(torch.arange(z.shape[-1],device=z.device,dtype=z.dtype)/z.shape[-1])))
            causal=torch.triu(torch.ones(z.shape[1],z.shape[1],device=z.device,dtype=torch.bool),diagonal=1)
            z=self.attention(z,mask=causal,src_key_padding_mask=padding)
        z=z[:,-1]
        if self.kind=='tft':
            z=z*torch.sigmoid(self.output_gate(z))
        return ordered_head(self.output(z))


class RGMF(nn.Module):
    """Joint variant: ordered source experts with the same convex gate for every q."""
    def __init__(self,base_features,source_features,meta_features,width=32,kind='gru'):
        super().__init__()
        self.base=Encoder(base_features,width,kind)
        self.experts=nn.ModuleList([Encoder(base_features+n,width,kind) for n in source_features])
        self.gate=nn.Linear(meta_features,len(source_features)+1)

    def forward(self,base,sources,meta,valid):
        mu0,q0=self.base(base)
        mus=[mu0]
        qs=[q0]
        for expert,source in zip(self.experts,sources):
            # Masked NaNs are sanitized before expert arithmetic, avoiding 0*NaN.
            mu,q=expert(torch.cat([base,torch.nan_to_num(source)],dim=-1))
            mus.append(mu)
            qs.append(q)
        valid=valid.bool()
        mask=torch.cat([torch.ones(len(valid),1,dtype=torch.bool,device=valid.device),valid],dim=1)
        logits=self.gate(meta).masked_fill(~mask,float('-inf'))
        gates=torch.softmax(logits,dim=1)
        mu=(torch.stack(mus,dim=1)*gates).sum(dim=1)
        q=(torch.stack(qs,dim=1)*gates[:,:,None]).sum(dim=1)
        all_missing=~valid.any(dim=1)
        # Explicit selection ensures exact base fallback even in edge numerical cases.
        mu=torch.where(all_missing,mu0,mu)
        q=torch.where(all_missing[:,None],q0,q)
        return mu,q,gates


def normalized_loss(y,mean,quantiles,scale,weights=None,alpha=1):
    y,mean,quantiles=y.float(),mean.float(),quantiles.float()
    q=torch.tensor([.05,.1,.5,.9,.95],device=y.device)
    scale=torch.as_tensor(scale,device=y.device,dtype=torch.float32)
    if not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Positive shared train scale required')
    err=y[:,None]-quantiles
    pin=torch.maximum(q*err,(q-1)*err).mean(dim=1)/scale
    losses=pin+alpha*((y-mean)/scale)**2
    if weights is None:
        return losses.mean()
    w=weights.float()
    if w.sum()<=0 or (w<0).any():
        raise ValueError('Invalid loss weights')
    return (losses*w).sum()/w.sum()


def require_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('BLOCKED_GPU: actual single CUDA required')
    if '4090 Laptop' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Unqualified GPU identity')


def fit_encoder(model,x,y,scale,epochs=100,lr=.003,seed=11,weights=None,observed=None,elapsed=None):
    require_cuda()
    torch.manual_seed(seed)
    model=model.cuda()
    x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
    y=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    if not len(y) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Invalid neural data')
    kwargs={}
    if observed is not None:
        kwargs['observed']=torch.as_tensor(observed,dtype=torch.bool,device='cuda')
        kwargs['elapsed']=torch.as_tensor(elapsed,dtype=torch.float32,device='cuda')
    w=None if weights is None else torch.as_tensor(weights,dtype=torch.float32,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    losses=[]
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mean,q=model(x,**kwargs)
        loss=normalized_loss(y,mean,q,scale,w)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradients')
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model,losses
