"""Multi-asset residual fusion with past-only native-unit prequential baselines."""
import numpy as np
import pandas as pd
import torch
from .track_statistics import DateWeightedStatistical as Statistical
from .features import TrainTransform
from .neural import require_cuda,normalized_loss
from .residual import ResidualFusion
from .track_neural import raw_source_validity
from .runtime import digest


def native_oof_base(raw,y,assets,dates,ends,available,base_columns,minimum=26,block=13):
    raw=np.asarray(raw);y=np.asarray(y,dtype=float);assets=np.asarray(assets)
    dates=pd.DatetimeIndex(pd.to_datetime(dates,utc=True,format='mixed'))
    ends=pd.DatetimeIndex(pd.to_datetime(ends,utc=True,format='mixed'));available=pd.DatetimeIndex(pd.to_datetime(available,utc=True,format='mixed'))
    if len(raw)!=len(y) or len(assets)!=len(y) or not np.isfinite(y).all() or (available<ends).any() or (ends<=dates).any():raise ValueError('Complete mature native OOF targets required')
    unique=dates.unique().sort_values();mean=np.full(len(y),np.nan);quantiles=np.full((len(y),5),np.nan)
    for start in range(minimum,len(unique),block):
        cutoff=unique[start];train=(dates<cutoff)&(ends<=cutoff)&(available<=cutoff);test=dates.isin(unique[start:start+block])
        if dates[train].nunique()<minimum:continue
        scales={asset:max(float(y[train&(assets==asset)].std()),1e-8) for asset in np.unique(assets[train])}
        if not set(assets[test])<=set(scales):continue
        tx=raw[train][...,base_columns].reshape(train.sum(),-1);vx=raw[test][...,base_columns].reshape(test.sum(),-1)
        transform=TrainTransform().fit(tx,dates[train],cutoff)
        weights=1/pd.Series(dates[train]).map(pd.Series(dates[train]).value_counts()).to_numpy()
        train_scale=np.asarray([scales[a] for a in assets[train]]);test_scale=np.asarray([scales[a] for a in assets[test]])
        model=Statistical('ridge').fit(transform.transform(tx),y[train]/train_scale,weights=weights)
        mu,q=model.predict(transform.transform(vx));mean[test]=mu*test_scale;quantiles[test]=q*test_scale[:,None]
    return mean,quantiles


class TrackResidualCUDA:
    device='cuda:0'
    def __init__(self,contract,width=32,lr=.003,epochs=100,seed=11):
        if not contract['source_raw_columns']:raise ValueError('Residual fusion requires additional registered sources')
        self.config={'contract':contract,'width':width,'lr':lr,'epochs':epochs,'seed':seed}

    def fit(self,raw,training,scales,cutoff,checkpoint_path=None,checkpoint_seconds=120):
        require_cuda();c=self.config;contract=c['contract'];torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        cutoff=pd.Timestamp(cutoff);known=pd.to_datetime(training.max_dependency_available_at,utc=True)
        dates=pd.to_datetime(training.decision_time,utc=True);available=pd.to_datetime(training.label_available_at,utc=True)
        if cutoff.tzinfo is None or (known>dates).any() or (available>cutoff).any() or (dates>cutoff).any():raise PermissionError('Residual future dependency/cutoff')
        raw=np.asarray(raw,dtype=np.float32);outer_scale=training.asset.map(scales).to_numpy(dtype=float)
        y=training.y.to_numpy(dtype=float)
        native_mean,native_q=native_oof_base(raw,y,training.asset,training.decision_time,training.label_end,training.label_available_at,contract['base_raw_columns'])
        usable=np.isfinite(native_mean)&np.isfinite(native_q).all(1)
        if dates[usable].nunique()<13:raise ValueError('Insufficient distinct mature OOF dates')
        base_raw=raw[...,contract['base_raw_columns']].reshape(len(raw),-1)
        self.base_transform=TrainTransform().fit(base_raw,dates,cutoff)
        weights=1/training.groupby('decision_time').asset.transform('count').to_numpy()
        self.base=Statistical('ridge').fit(self.base_transform.transform(base_raw),y/outer_scale,weights=weights)
        self.transform=TrainTransform().fit(raw.reshape(-1,raw.shape[-1]),np.repeat(dates.to_numpy(),raw.shape[1]),cutoff)
        full=self.transform.transform(raw.reshape(-1,raw.shape[-1])).reshape(len(raw),raw.shape[1],-1)
        self.raw_dimension=raw.shape[-1];expand=lambda columns:columns+[i+self.raw_dimension for i in columns]
        self.base_columns=expand(contract['base_raw_columns']);self.source_columns=[expand(v) for v in contract['source_raw_columns']];self.meta_columns=expand(contract['meta_raw_columns'])
        self.model=ResidualFusion([len(self.base_columns)+len(v) for v in self.source_columns],len(self.meta_columns),c['width'],'gru').cuda()
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        if self.parameter_count>500000:raise ValueError('Core residual parameter cap exceeded')
        tx=torch.as_tensor(full[usable],dtype=torch.float32,device='cuda');ty=torch.as_tensor(y[usable]/outer_scale[usable],dtype=torch.float32,device='cuda')
        base_mu=torch.as_tensor(native_mean[usable]/outer_scale[usable],dtype=torch.float32,device='cuda')
        base_q=torch.as_tensor(native_q[usable]/outer_scale[usable,None],dtype=torch.float32,device='cuda')
        valid=torch.as_tensor(raw_source_validity(raw[usable],contract['source_value_columns']),device='cuda')
        args=([tx[...,self.base_columns+v] for v in self.source_columns],tx[:,-1,self.meta_columns],valid)
        weight=torch.as_tensor(weights[usable],dtype=torch.float32,device='cuda');optimizer=torch.optim.AdamW(self.model.parameters(),lr=c['lr']);self.losses=[]
        self.oof_dates=int(dates[usable].nunique());self.oof_id=digest({'mean':native_mean[usable].tolist(),'q':native_q[usable].tolist(),'dates':dates[usable].astype(str).tolist()})
        import time
        start_epoch=0;last_checkpoint=time.monotonic()
        if checkpoint_path is not None:
            from .checkpoint import load_checkpoint,save_checkpoint,restore_rng
            identity=digest({'config':c,'cutoff':cutoff.isoformat(),'transform':self.transform.manifest(),'oof_id':self.oof_id,
                'native_targets':y.tolist(),'assets':training.asset.tolist(),'outer_scales':outer_scale.tolist(),'weights':weights.tolist(),
                'label_available_at':available.astype(str).tolist()})
            state=load_checkpoint(checkpoint_path,identity)
            if state is not None:
                self.model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state)
                start_epoch=state['step'];self.losses=state['losses']
                if state['sample_cursor']!=start_epoch*int(usable.sum()):raise ValueError('Residual checkpoint sample accounting mismatch')
        for epoch in range(start_epoch,c['epochs']):
            optimizer.zero_grad(set_to_none=True);mu,q,_=self.model(base_mu,base_q,*args);loss=normalized_loss(ty,mu,q,1.,weight)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite multi-asset residual loss')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite residual gradients')
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);optimizer.step();self.losses.append(float(loss.detach()))
            if checkpoint_path is not None and (time.monotonic()-last_checkpoint>=checkpoint_seconds or epoch+1==c['epochs']):
                save_checkpoint(checkpoint_path,{'model':self.model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':None,
                    'step':epoch+1,'sample_cursor':(epoch+1)*int(usable.sum()),'losses':self.losses},identity);last_checkpoint=time.monotonic()
        self.model.eval()
        return self

    def predict(self,raw):
        require_cuda();raw=np.asarray(raw,dtype=float)
        mean,q=self.base.predict(self.base_transform.transform(raw[...,self.config['contract']['base_raw_columns']].reshape(len(raw),-1)))
        full=self.transform.transform(raw.reshape(-1,raw.shape[-1])).reshape(len(raw),raw.shape[1],-1)
        with torch.no_grad():
            tx=torch.as_tensor(full,dtype=torch.float32,device='cuda');valid=torch.as_tensor(raw_source_validity(raw,self.config['contract']['source_value_columns']),device='cuda')
            mu,pq,_=self.model(torch.as_tensor(mean,dtype=torch.float32,device='cuda'),torch.as_tensor(q,dtype=torch.float32,device='cuda'),
                [tx[...,self.base_columns+v] for v in self.source_columns],tx[:,-1,self.meta_columns],valid)
        return mu.cpu().numpy(),pq.cpu().numpy()

    def save(self,path):
        torch.save({'config':self.config,'base':self.base,'base_transform':self.base_transform,'transform':self.transform,'state':self.model.state_dict(),
            'raw_dimension':self.raw_dimension,'base_columns':self.base_columns,'source_columns':self.source_columns,'meta_columns':self.meta_columns,
            'losses':self.losses,'oof_dates':self.oof_dates,'oof_id':self.oof_id},path)

    @staticmethod
    def load(path):
        require_cuda();saved=torch.load(path,map_location='cuda',weights_only=False);model=TrackResidualCUDA(**saved['config'])
        for key in ['base','base_transform','transform','raw_dimension','base_columns','source_columns','meta_columns','losses','oof_dates','oof_id']:setattr(model,key,saved[key])
        model.model=ResidualFusion([len(model.base_columns)+len(v) for v in model.source_columns],len(model.meta_columns),model.config['width'],'gru').cuda()
        model.model.load_state_dict(saved['state']);model.model.eval();return model
