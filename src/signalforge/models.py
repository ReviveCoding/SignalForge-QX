"""Statistical/CUDA tree adapters; all predictions have separate means and quantiles."""
import pickle
from pathlib import Path
import numpy as np

QUANTILES = np.array([.05,.1,.5,.9,.95])


class Statistical:
    device = 'cpu'

    def __init__(self, family='ridge', regularization=1.0):
        if family not in {'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}:
            raise ValueError('Unknown statistical family')
        self.family,self.regularization = family,regularization

    def fit(self,x,y,weights=None):
        from sklearn.linear_model import Ridge, QuantileRegressor
        x,y=np.asarray(x),np.asarray(y)
        if x.ndim!=2 or y.shape!=(len(x),) or len(y)==0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid fit data')
        w=np.ones(len(y)) if weights is None else np.asarray(weights)
        if w.shape!=y.shape or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
            raise ValueError('Invalid training weights')
        if self.family in {'historical','ewma'}:
            if self.family=='ewma':
                w=w*.97**np.arange(len(y)-1,-1,-1)
            self.mean=float(np.average(y,weights=w))
            order=np.argsort(y)
            cum=np.cumsum(w[order])/w.sum()
            self.q=np.interp(QUANTILES,cum,y[order])
        else:
            # Mixed-frequency challenger is a shrinkage distributed-lag regression;
            # it is not represented as a Kalman/state-space reproduction.
            self.mean_model=Ridge(alpha=self.regularization).fit(x,y,sample_weight=w)
            if self.family=='linear_quantile':
                self.qmodels=[QuantileRegressor(quantile=q,alpha=self.regularization,solver='highs').fit(x,y,sample_weight=w) for q in QUANTILES]
            else:
                self.qmodels=None
                self.residual_q=np.quantile(y-self.mean_model.predict(x),QUANTILES)
        return self

    def predict(self,x):
        x=np.asarray(x)
        if self.family in {'historical','ewma'}:
            return np.full(len(x),self.mean),np.tile(self.q,(len(x),1))
        mean=self.mean_model.predict(x)
        q=np.column_stack([m.predict(x) for m in self.qmodels]) if self.qmodels else mean[:,None]+self.residual_q
        return mean,np.sort(q,axis=1)

    def save(self,path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        """Only load verified local artifacts; pickle is not a public-data parser."""
        return pickle.loads(Path(path).read_bytes())


class CudaTree:
    device='cuda:0'

    def __init__(self,family='lightgbm',rounds=100,seed=11,regularization=1.0):
        if family not in {'lightgbm','xgboost'}:
            raise ValueError('Unknown tree family')
        self.family,self.rounds,self.seed,self.regularization=family,rounds,seed,regularization

    def fit(self,x,y,weights=None):
        # Caller owns shared GPU lease for the entire fit/inference lifecycle.
        x,y=np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)
        if x.ndim!=2 or y.shape!=(len(x),) or not len(y) or not np.isfinite(y).all() or np.isinf(x).any():
            raise ValueError('Invalid CUDA tree fit data')
        if self.family=='lightgbm':
            import lightgbm as lgb
            params={'device_type':'cuda','num_threads':4,'verbosity':-1,'seed':self.seed,'num_leaves':15,
                    'max_bin':63,'min_data_in_leaf':10,'lambda_l2':self.regularization}
            self.models=[]
            for q in [None,*QUANTILES]:
                p={**params,'objective':'regression' if q is None else 'quantile'}
                if q is not None:
                    p['alpha']=float(q)
                model=lgb.train(p,lgb.Dataset(x,label=y,weight=weights),num_boost_round=self.rounds)
                if model.params.get('device_type')!='cuda':
                    raise RuntimeError('CUDA fallback prohibited')
                self.models.append(model)
        else:
            import xgboost as xgb
            self.models=[]
            for q in [None,QUANTILES.tolist()]:
                p={'device':'cuda:0','tree_method':'hist','max_depth':3,'seed':self.seed,'nthread':4,
                   'lambda':self.regularization,'objective':'reg:squarederror' if q is None else 'reg:quantileerror'}
                if q is not None:
                    p['quantile_alpha']=q
                m=xgb.train(p,xgb.QuantileDMatrix(x,label=y,weight=weights),num_boost_round=self.rounds)
                import json
                if not json.loads(m.save_config())['learner']['generic_param']['device'].startswith('cuda'):
                    raise RuntimeError('XGBoost CUDA fallback prohibited')
                self.models.append(m)
        return self

    def predict(self,x):
        x=np.asarray(x,dtype=np.float32)
        if self.family=='lightgbm':
            return self.models[0].predict(x),np.sort(np.column_stack([m.predict(x) for m in self.models[1:]]),axis=1)
        import xgboost as xgb
        d=xgb.DMatrix(x)
        return self.models[0].predict(d),np.sort(self.models[1].predict(d),axis=1)

    def save(self,path):
        Path(path).write_bytes(pickle.dumps(self))

    load=staticmethod(Statistical.load)


def prequential_base(x,y,dates,min_train_dates=26,block_dates=13,*,label_available_at=None,label_end=None):
    """Two-stage residual inputs are never in-sample fitted base forecasts."""
    dates=np.asarray(dates)
    if label_available_at is None or label_end is None:
        raise ValueError('Actual label publication and interval end required for prequential fitting')
    available,end=np.asarray(label_available_at),np.asarray(label_end)
    if dates.shape!=available.shape or dates.shape!=end.shape or len(dates)!=len(y):
        raise ValueError('Aligned temporal metadata required')
    if (end<=dates).any() or (available<end).any():
        raise ValueError('Invalid label maturity chronology')
    unique=np.unique(dates)
    mean=np.full(len(y),np.nan)
    quantiles=np.full((len(y),5),np.nan)
    for start in range(min_train_dates,len(unique),block_dates):
        train=(dates<unique[start])&(available<=unique[start])&(end<=unique[start])
        test=np.isin(dates,unique[start:start+block_dates])
        if len(np.unique(dates[train]))<min_train_dates:
            continue
        model=Statistical('ridge').fit(np.asarray(x)[train],np.asarray(y)[train])
        mean[test],quantiles[test]=model.predict(np.asarray(x)[test])
    return mean,quantiles


class NeuralCUDA:
    """CUDA-only joint adapter, with explicit compact architecture/method metadata."""
    device='cuda:0'

    def __init__(self,kind='gru',width=32,epochs=100,seed=11,lr=.003,base_columns=None,source_columns=None,meta_columns=None):
        self.config={'kind':kind,'width':width,'epochs':epochs,'seed':seed,'lr':lr,
                     'base_columns':base_columns,'source_columns':source_columns,'meta_columns':meta_columns}

    def build(self,features):
        from .neural import Encoder,RGMF
        c=self.config
        if c['kind'].startswith('rgmf_'):
            if not c['base_columns'] or not c['source_columns'] or not c['meta_columns']:
                raise ValueError('Registered base/source/meta columns required')
            model=RGMF(len(c['base_columns']),[len(v) for v in c['source_columns']],len(c['meta_columns']),
                       c['width'],c['kind'].removeprefix('rgmf_'))
        else:model=Encoder(features,c['width'],c['kind'])
        if sum(p.numel() for p in model.parameters())>500000:
            raise ValueError('Core parameter cap exceeded')
        return model

    def forward(self,x,observed=None,elapsed=None,source_valid=None):
        import torch
        c=self.config
        if c['kind']=='mlp' and x.ndim==3:
            x=x.reshape(len(x),-1)
        if c['kind'].startswith('rgmf_'):
            base=x[...,c['base_columns']]
            sources=[x[...,v] for v in c['source_columns']]
            latest=x[:,-1] if x.ndim==3 else x
            meta=latest[:,c['meta_columns']]
            # Explicit caller mask defines source validity; imputed zeros are never availability.
            if observed is None:raise ValueError('Source validity masks required for RGMF')
            latest_mask=observed[:,-1] if observed.ndim==3 else observed
            valid=torch.stack([latest_mask[:,v].any(dim=1) for v in c['source_columns']],dim=1) if source_valid is None else source_valid
            if valid.shape!=(len(x),len(c['source_columns'])):raise ValueError('Source validity shape mismatch')
            mean,q,_=self.model(base,sources,meta,valid)
            return mean,q
        if c['kind']=='gru_d':return self.model(x,observed=observed,elapsed=elapsed)
        return self.model(x)

    def fit(self,x,y,scale,weights=None,observed=None,elapsed=None,validation=None,checkpoint_path=None,checkpoint_seconds=120,
            initial_encoder_state=None,initialization_id=None):
        import torch
        from .neural import normalized_loss,require_cuda
        require_cuda();torch.manual_seed(self.config['seed']);torch.cuda.manual_seed_all(self.config['seed'])
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        if x.ndim not in {2,3} or not len(y) or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid neural training sample')
        self.features=(x.shape[-1]*x.shape[1] if self.config['kind']=='mlp' and x.ndim==3 else x.shape[-1])
        self.model=self.build(self.features).cuda()
        self.initialization_id=initialization_id
        if initial_encoder_state is not None:
            if not initialization_id or any(not k.startswith(('input.','sequence.')) for k in initial_encoder_state):
                raise ValueError('Registered encoder-only initialization required')
            self.model.load_state_dict(initial_encoder_state,strict=False)
        tx=torch.as_tensor(x,device='cuda');ty=torch.as_tensor(y,device='cuda')
        tw=None if weights is None else torch.as_tensor(weights,dtype=torch.float32,device='cuda')
        mask=None if observed is None else torch.as_tensor(observed,dtype=torch.bool,device='cuda')
        age=None if elapsed is None else torch.as_tensor(elapsed,dtype=torch.float32,device='cuda')
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.config['lr'])
        self.losses=[];self.validation_losses=[];best=float('inf');best_state=None
        start_epoch=0
        import time
        last_checkpoint=time.monotonic()
        if checkpoint_path is not None:
            from .runtime import digest
            from .checkpoint import load_checkpoint,save_checkpoint,restore_rng
            identity=digest({'config':self.config,'initialization_id':initialization_id,'x':x.tolist(),'y':y.tolist(),'scale':np.asarray(scale).tolist(),
                             'observed':None if observed is None else np.asarray(observed).tolist(),
                             'elapsed':None if elapsed is None else np.asarray(elapsed).tolist(),
                             'weights':None if weights is None else np.asarray(weights).tolist(),
                             'validation':None if validation is None else [None if v is None else np.asarray(v).tolist() for v in validation]})
            state=load_checkpoint(checkpoint_path,identity)
            if state is not None:
                self.model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state)
                start_epoch=state['step'];self.losses=state['losses'];self.validation_losses=state['validation_losses']
                best=state['best'];best_state=state['best_state']
                if state['sample_cursor']!=start_epoch*len(y):raise ValueError('Checkpoint sample accounting mismatch')
        for epoch in range(start_epoch,self.config['epochs']):
            self.model.train();optimizer.zero_grad(set_to_none=True)
            mean,q=self.forward(tx,mask,age);loss=normalized_loss(ty,mean,q,scale,tw)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural loss')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite neural gradients')
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);optimizer.step()
            self.losses.append(float(loss.detach()))
            if validation is not None:
                # Caller must supply only registered inner validation; outer labels never enter this adapter.
                vx,vy,vm,va=validation
                self.model.eval()
                with torch.no_grad():
                    args=[torch.as_tensor(vx,dtype=torch.float32,device='cuda'),
                          None if vm is None else torch.as_tensor(vm,dtype=torch.bool,device='cuda'),
                          None if va is None else torch.as_tensor(va,dtype=torch.float32,device='cuda')]
                    score=float(normalized_loss(torch.as_tensor(vy,dtype=torch.float32,device='cuda'),*self.forward(*args),scale))
                self.validation_losses.append(score)
                if score<best:
                    best=score;best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            if checkpoint_path is not None and (time.monotonic()-last_checkpoint>=checkpoint_seconds or epoch+1==self.config['epochs']):
                save_checkpoint(checkpoint_path,{'model':self.model.state_dict(),'optimizer':optimizer.state_dict(),
                                'scheduler':None,'step':epoch+1,'sample_cursor':(epoch+1)*len(y),'losses':self.losses,
                                'validation_losses':self.validation_losses,'best':best,'best_state':best_state},identity)
                last_checkpoint=time.monotonic()
        if best_state is not None:self.model.load_state_dict(best_state)
        self.model.eval();self.parameter_count=sum(p.numel() for p in self.model.parameters())
        return self

    def predict(self,x,observed=None,elapsed=None,source_valid=None):
        import torch
        from .neural import require_cuda
        require_cuda();self.model.eval()
        with torch.no_grad():
            mean,q=self.forward(torch.as_tensor(x,dtype=torch.float32,device='cuda'),
                                None if observed is None else torch.as_tensor(observed,dtype=torch.bool,device='cuda'),
                                None if elapsed is None else torch.as_tensor(elapsed,dtype=torch.float32,device='cuda'),
                                None if source_valid is None else torch.as_tensor(source_valid,dtype=torch.bool,device='cuda'))
        return mean.cpu().numpy(),q.cpu().numpy()

    def save(self,path):
        import torch
        torch.save({'config':self.config,'features':self.features,'model':self.model.state_dict(),
                    'method':'joint','initialization_id':getattr(self,'initialization_id',None),'losses':self.losses,'validation_losses':self.validation_losses},path)

    @staticmethod
    def load(path):
        import torch
        from .neural import require_cuda
        require_cuda();state=torch.load(path,map_location='cuda',weights_only=False)
        adapter=NeuralCUDA(**state['config']);adapter.features=state['features']
        adapter.model=adapter.build(adapter.features).cuda();adapter.model.load_state_dict(state['model']);adapter.model.eval()
        adapter.losses=state['losses'];adapter.validation_losses=state['validation_losses']
        adapter.initialization_id=state.get('initialization_id')
        return adapter
