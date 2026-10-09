"""Synthetic BAR/ABC engineering prototypes, NOT trained or qualified research models."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from .pit import _aware_utc

def mature_partitions(decisions,ends,available,minimum=26,block=13):
    dates=pd.DatetimeIndex(_aware_utc(decisions,'OOF decisions'))
    ends=pd.DatetimeIndex(_aware_utc(ends,'label ends')); available=pd.DatetimeIndex(_aware_utc(available,'label availability'))
    if len(dates)!=len(ends) or len(dates)!=len(available) or minimum<1 or block<1:
        raise ValueError('Aligned chronological partition metadata')
    if len(dates) and dates.max()>=pd.Timestamp('2024-01-01T00:00Z'):
        raise PermissionError('Reserved OOF forbidden')
    if (ends<=dates).any() or (available<ends).any():raise ValueError('Invalid maturity')
    result=[]; unique=dates.unique().sort_values()
    for start in range(minimum,len(unique),block):
        cutoff=unique[start]; train=np.flatnonzero((dates<cutoff)&(ends<=cutoff)&(available<=cutoff))
        evaluation=np.flatnonzero(dates.isin(unique[start:start+block]))
        if dates[train].nunique()>=minimum:
            result.append({'cutoff':cutoff.isoformat(),'train':train,'oof':evaluation})
    return result

def select_inner_baseline(trials):
    if not trials or any(t.get('partition')!='inner_validation' or
        not np.isfinite(t.get('normalized_pinball',np.nan)) or not t.get('baseline_id') for t in trials):
        raise PermissionError('Inner-only qualified baseline selection required')
    return min(trials,key=lambda t:(t['normalized_pinball'],t['baseline_id']))['baseline_id']

def _base(mean,q):
    if mean.ndim!=1 or q.shape!=(len(mean),5) or not torch.isfinite(mean).all() or not torch.isfinite(q).all() or (q[:,1:]<q[:,:-1]).any():
        raise ValueError('Finite ordered normalized frozen baseline required')

def _weights(gates,valid,ood):
    if valid.dtype!=torch.bool or ood.dtype!=torch.bool or valid.shape!=ood.shape or gates.shape!=valid.shape:
        raise ValueError('Explicit typed source availability/OOD required')
    if not torch.isfinite(gates).all() or (gates<0).any() or (gates.sum(1)>1+1e-6).any():
        raise ValueError('Nonnegative source gate mass <=1 required')
    return gates*(valid&~ood)

class BaselineAnchoredResidual(nn.Module):
    def __init__(self,source_dimensions,bound=.25):
        super().__init__()
        if not source_dimensions or not 0<bound<=.25:raise ValueError('Registered bounded prototype')
        self.bound=bound
        self.heads=nn.ModuleList([nn.Linear(n,6) for n in source_dimensions])
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,base_mean,base_q,sources,gates,valid,ood):
        _base(base_mean,base_q); w=_weights(gates,valid,ood)
        if len(sources)!=len(self.heads) or w.shape!=(len(base_mean),len(sources)):
            raise ValueError('Aligned source heads')
        values=[]
        for i,(head,x) in enumerate(zip(self.heads,sources)):
            if x.shape!=(len(base_mean),head.in_features):raise ValueError('Source shape')
            active=valid[:,i]&~ood[:,i]
            if not torch.isfinite(x[active]).all():raise ValueError('Nonfinite valid source')
            clean=torch.where(active[:,None],x,torch.zeros_like(x))
            values.append(self.bound*torch.tanh(head(clean)))
        delta=(torch.stack(values,1)*w[:,:,None]).sum(1)
        # Isotonic monotone projection: baseline itself is unchanged at zero delta.
        q=torch.cummax(base_q+delta[:,1:],dim=1).values
        return base_mean+delta[:,0],q
    def manifest(self):
        return {'prototype':'sgqx-v3.3-BAR','baseline_units':'shared_train_scale_normalized',
                'residual_bound':self.bound,'zero_init_last_head':True,
                'baseline_selection':'inner_only','residual_fit_evidence':'mature_prequential_OOF',
                'real_fit_executed':False,'qualified_for_final':False}

def asymmetric_quantiles(base_q,rho,deflate,inflate,shift,gates,valid,ood):
    _base(base_q[:,2],base_q)
    w=_weights(gates,valid,ood); n,s=w.shape
    if any(a.shape!=(n,s) for a in [rho,deflate,inflate,shift]):
        raise ValueError('Aligned source coefficients')
    if not all(torch.isfinite(a).all() for a in [rho,deflate,inflate,shift]) or (rho<0).any() or (rho>1).any() or (deflate<0).any() or (deflate>inflate).any() or (inflate>1).any() or (shift.abs()>.25).any():
        raise ValueError('ABC coefficient/order/bound constraints')
    middle=base_q[:,2,None]+shift
    lower=base_q[:,2,None,None]-base_q[:,None,:2]
    upper=base_q[:,None,3:]-base_q[:,2,None,None]
    corrected=torch.cat([middle[:,:,None]-lower*(1-rho*deflate)[:,:,None],
                         middle[:,:,None],
                         middle[:,:,None]+upper*(1+rho*inflate)[:,:,None]],dim=-1)
    # Convex mix of baseline and individually ordered source corrections.
    return base_q+(w[:,:,None]*(corrected-base_q[:,None,:])).sum(1)

class AsymmetricSourceCorrection(nn.Module):
    def __init__(self,source_dimensions):
        super().__init__(); self.heads=nn.ModuleList([nn.Linear(n,4) for n in source_dimensions])
        if not source_dimensions:raise ValueError('Sources required')
        for h in self.heads:nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    def forward(self,base_q,sources,gates,valid,ood):
        _base(base_q[:,2],base_q); _weights(gates,valid,ood)
        if len(sources)!=len(self.heads) or gates.shape!=(len(base_q),len(sources)):raise ValueError('Aligned ABC sources')
        raw=[]
        for i,(head,x) in enumerate(zip(self.heads,sources)):
            active=valid[:,i]&~ood[:,i]
            if x.shape!=(len(base_q),head.in_features) or not torch.isfinite(x[active]).all():raise ValueError('Invalid valid ABC source')
            raw.append(head(torch.where(active[:,None],x,torch.zeros_like(x))))
        a=torch.stack(raw,dim=1); shift=.25*torch.tanh(a[:,:,0])
        rho=torch.clamp(2*torch.sigmoid(a[:,:,1])-1,0,1)
        deflate=.5*torch.sigmoid(a[:,:,2]); inflate=deflate+.5*torch.sigmoid(a[:,:,3])
        return asymmetric_quantiles(base_q,rho,deflate,inflate,shift,gates,valid,ood)
    def manifest(self):
        return {'prototype':'sgqx-v3.3-ABC','rho_range':[0,1],'deflate_le_inflate':True,
                'inflate_cap':1,'shift_bound':.25,'zero_init_identity':True,
                'tail_fit_window':'mature_training_OOF_only','eval_labels_accepted':False,
                'real_fit_executed':False,'qualified_for_final':False}
