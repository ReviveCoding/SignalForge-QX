"""CUDA two-stage RGMF, with mature temporal OOF and a frozen CPU base.

Train transforms are refitted within each prequential block. The adapter never
accepts in-sample base predictions as residual training evidence.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .models import Statistical, QUANTILES
from .features import TrainTransform
from .neural import Encoder, require_cuda, normalized_loss
from .pit import _aware_utc


def mature_oof(x, y, decisions, label_ends, label_available, minimum=26, block=13):
    dates=pd.DatetimeIndex(_aware_utc(decisions,'OOF decisions'))
    ends=pd.DatetimeIndex(_aware_utc(label_ends,'OOF label ends'));available=pd.DatetimeIndex(_aware_utc(label_available,'OOF label publication'))
    if len(x)!=len(y) or len(dates)!=len(y) or len(ends)!=len(y) or len(available)!=len(y):
        raise ValueError('Aligned OOF metadata required')
    if (ends<=dates).any() or (available<ends).any():
        raise ValueError('Invalid label chronology')
    x=np.asarray(x);y=np.asarray(y)
    unique=dates.unique().sort_values();mu=np.full(len(y),np.nan);q=np.full((len(y),5),np.nan)
    for start in range(minimum,len(unique),block):
        origin=unique[start]
        train=(dates<origin)&(ends<=origin)&(available<=origin)
        test=dates.isin(unique[start:start+block])
        if len(dates[train].unique())<minimum:continue
        tx=x[train].reshape(len(x[train]),-1);vx=x[test].reshape(len(x[test]),-1)
        transform=TrainTransform().fit(tx,dates[train],origin)
        base=Statistical('ridge').fit(transform.transform(tx),y[train])
        mu[test],q[test]=base.predict(transform.transform(vx))
    return mu,q


class ResidualFusion(nn.Module):
    def __init__(self, features, meta_features, width=32, kind='gru'):
        super().__init__()
        self.experts=nn.ModuleList([Encoder(n,width,kind) for n in features])
        self.gate=nn.Linear(meta_features,len(features)+1)

    def forward(self, base_mu, base_q, sources, meta, valid):
        means=[base_mu];quantiles=[base_q]
        for expert,source in zip(self.experts,sources):
            residual_mu,residual_q=expert(torch.nan_to_num(source))
            means.append(base_mu+residual_mu)
            quantiles.append(base_mu[:,None]+residual_q)
        mask=torch.cat([torch.ones(len(valid),1,dtype=torch.bool,device=valid.device),valid.bool()],1)
        gates=torch.softmax(self.gate(meta).masked_fill(~mask,float('-inf')),1)
        mu=(torch.stack(means,1)*gates).sum(1)
        q=(torch.stack(quantiles,1)*gates[:,:,None]).sum(1)
        absent=~valid.any(1)
        return torch.where(absent,base_mu,mu),torch.where(absent[:,None],base_q,q),gates


class ResidualRGMFCUDA:
    device='cuda:0'

    def __init__(self, source_columns, meta_columns, base_columns=None, kind='gru', width=32, epochs=100, seed=11, lr=.003):
        if not base_columns or not source_columns or not meta_columns:raise ValueError('Explicit registered base/source/meta columns required')
        if any(set(base_columns)&set(source) for source in source_columns):raise ValueError('Additional source cannot enter the frozen base')
        self.config=dict(source_columns=source_columns,meta_columns=meta_columns,base_columns=base_columns,kind=kind,width=width,epochs=epochs,seed=seed,lr=lr)

    def fit(self, x, y, scale, decisions, label_ends, label_available, cutoff, observed, minimum=26, block=13):
        require_cuda();c=self.config;torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        dates=pd.DatetimeIndex(_aware_utc(decisions,'fit decisions'));available=pd.DatetimeIndex(_aware_utc(label_available,'fit label publication'))
        ends=pd.DatetimeIndex(_aware_utc(label_ends,'fit label ends'));cut=pd.Timestamp(cutoff)
        if cut.tzinfo is None or (dates>cut).any() or (available>cut).any() or (ends>cut).any():
            raise ValueError('All fit rows and labels must mature by cutoff')
        if observed.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Finite transformed inputs and explicit masks required')
        bx=x[...,c['base_columns']]
        oof_mu,oof_q=mature_oof(bx,y,dates,ends,available,minimum,block)
        usable=np.isfinite(oof_mu)&np.isfinite(oof_q).all(1)
        if usable.sum()<13:raise ValueError('Insufficient mature prequential residual rows')
        flat=bx.reshape(len(x),-1)
        self.transform=TrainTransform().fit(flat,dates,cut)
        self.base=Statistical('ridge').fit(self.transform.transform(flat),y)
        steps=x.shape[1] if x.ndim==3 else 1
        context_dates=np.repeat(dates.to_numpy(),steps)
        self.input_transform=TrainTransform().fit(x.reshape(-1,x.shape[-1]),context_dates,cut)
        self.model=ResidualFusion([len(c['base_columns'])+len(v) for v in c['source_columns']],len(c['meta_columns']),c['width'],c['kind']).cuda()
        if sum(p.numel() for p in self.model.parameters())>500000:raise ValueError('Core parameter cap exceeded')
        sx=self.input_transform.transform(x.reshape(-1,x.shape[-1]),include_mask=False).reshape(x.shape)
        tx=torch.as_tensor(sx[usable],dtype=torch.float32,device='cuda');ty=torch.as_tensor(y[usable],device='cuda')
        mu=torch.as_tensor(oof_mu[usable],dtype=torch.float32,device='cuda')
        q=torch.as_tensor(oof_q[usable],dtype=torch.float32,device='cuda')
        args=self.inputs(tx,torch.as_tensor(observed[usable],dtype=torch.bool,device='cuda'))
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=c['lr']);self.losses=[]
        for epoch in range(c['epochs']):
            optimizer.zero_grad(set_to_none=True)
            pred_mean,pred_q,_=self.model(mu,q,*args)
            train_scale=np.asarray(scale)
            loss=normalized_loss(ty,pred_mean,pred_q,train_scale if train_scale.ndim==0 else train_scale[usable])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite residual loss')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite residual gradient')
            nn.utils.clip_grad_norm_(self.model.parameters(),1.);optimizer.step();self.losses.append(float(loss.detach()))
        self.model.eval();self.oof_count=int(usable.sum());self.method='two_stage_mature_prequential'
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        return self

    def inputs(self,x,observed):
        c=self.config;latest=x[:,-1] if x.ndim==3 else x
        mask=observed[:,-1] if observed.ndim==3 else observed
        return [x[...,c['base_columns']+v] for v in c['source_columns']],latest[:,c['meta_columns']],torch.stack([mask[:,v].any(1) for v in c['source_columns']],1)

    def predict(self,x,observed):
        require_cuda();x=np.asarray(x,dtype=np.float32)
        mu,q=self.base.predict(self.transform.transform(x[...,self.config['base_columns']].reshape(len(x),-1)))
        sx=self.input_transform.transform(x.reshape(-1,x.shape[-1]),include_mask=False).reshape(x.shape)
        with torch.no_grad():
            pm,pq,_=self.model(torch.as_tensor(mu,dtype=torch.float32,device='cuda'),torch.as_tensor(q,dtype=torch.float32,device='cuda'),
                              *self.inputs(torch.as_tensor(sx,dtype=torch.float32,device='cuda'),torch.as_tensor(observed,dtype=torch.bool,device='cuda')))
        return pm.cpu().numpy(),pq.cpu().numpy()

    def save(self,path):
        torch.save({'config':self.config,'base':self.base,'transform':self.transform,'input_transform':self.input_transform,'state':self.model.state_dict(),
                    'losses':self.losses,'oof_count':self.oof_count,'method':self.method},path)

    @staticmethod
    def load(path):
        require_cuda();saved=torch.load(path,map_location='cuda',weights_only=False)
        adapter=ResidualRGMFCUDA(**saved['config']);c=adapter.config
        adapter.base=saved['base'];adapter.transform=saved['transform'];adapter.input_transform=saved['input_transform']
        adapter.model=ResidualFusion([len(c['base_columns'])+len(v) for v in c['source_columns']],len(c['meta_columns']),c['width'],c['kind']).cuda()
        adapter.model.load_state_dict(saved['state']);adapter.model.eval()
        adapter.losses=saved['losses'];adapter.oof_count=saved['oof_count'];adapter.method=saved['method']
        return adapter
