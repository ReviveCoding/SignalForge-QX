"""Explicit source masks for multi-source tracks, including training/checkpoints."""
import numpy as np
from .models import NeuralCUDA
from .runtime import digest


def raw_source_validity(raw_contexts,value_columns):
    raw=np.asarray(raw_contexts,dtype=float)
    if raw.ndim!=3 or value_columns is None or any(not columns for columns in value_columns):raise ValueError('Registered source VALUE columns required')
    if any(i<0 or i>=raw.shape[-1] for columns in value_columns for i in columns):raise ValueError('Source validity column outside schema')
    if np.isinf(raw).any():raise ValueError('Infinite raw source values')
    return np.stack([np.isfinite(raw[:,-1,columns]).any(axis=1) for columns in value_columns],axis=1) if value_columns else np.zeros((len(raw),0),dtype=bool)


def elapsed_from_context_clocks(raw_contexts,context_decision_times,origins=None):
    """Days since last raw observation using actual ordered context clocks.

    Leading padding has zero elapsed time. A never-observed feature starts at
    the first actual context clock; appended observation indicators use zero.
    """
    import pandas as pd
    raw=np.asarray(raw_contexts,dtype=float)
    if raw.ndim!=3 or np.isinf(raw).any() or len(context_decision_times)!=len(raw):
        raise ValueError('Aligned finite-or-missing contexts and clocks required')
    elapsed=np.zeros_like(raw)
    for row,times in enumerate(context_decision_times):
        if len(times)!=raw.shape[1]:raise ValueError('Full padded context clocks required')
        present=[i for i,t in enumerate(times) if t is not None]
        if not present or present!=list(range(present[0],raw.shape[1])):raise ValueError('Only leading context clock padding allowed')
        clocks=[pd.Timestamp(times[i]) for i in present]
        if any(t.tzinfo is None for t in clocks) or any(b<=a for a,b in zip(clocks,clocks[1:])):
            raise ValueError('Aware strictly ordered context clocks required')
        if origins is not None and clocks[-1]!=pd.Timestamp(origins[row]):raise ValueError('Context must end at actual origin')
        last=np.full(raw.shape[-1],clocks[0].timestamp())
        for index,clock in zip(present,clocks):
            elapsed[row,index]=(clock.timestamp()-last)/86400
            last=np.where(np.isfinite(raw[row,index]),clock.timestamp(),last)
    return np.concatenate([elapsed,np.zeros_like(elapsed)],axis=-1)


class ExplicitSourceCUDA(NeuralCUDA):
    """Reuse validated CUDA optimizers/architectures without inferring masks from ages."""
    def build(self,features):
        if self.config['kind'].startswith('rgmf_') and self.config['source_columns']==[]:
            from .neural import RGMF
            model=RGMF(len(self.config['base_columns']),[],len(self.config['meta_columns']),self.config['width'],self.config['kind'].removeprefix('rgmf_'))
            if sum(p.numel() for p in model.parameters())>500000:raise ValueError('Core parameter cap exceeded')
            return model
        return super().build(features)
    def fit(self,x,y,scale,source_valid=None,**kwargs):
        if kwargs.get('validation') is not None:raise ValueError('Track HPO uses separate inner validation with explicit source masks')
        rgmf=self.config['kind'].startswith('rgmf_')
        if rgmf:
            valid=np.asarray(source_valid)
            if valid.dtype!=bool or valid.shape!=(len(x),len(self.config['source_columns'])):
                raise ValueError('Explicit typed raw source mask required during training')
            self.config['raw_source_mask_id']=digest(valid.tolist())
            import torch
            from .neural import require_cuda
            require_cuda();self._training_source_valid=torch.as_tensor(valid,dtype=torch.bool,device='cuda')
        try:return super().fit(x,y,scale,**kwargs)
        finally:self._training_source_valid=None

    def forward(self,x,observed=None,elapsed=None,source_valid=None):
        if self.config['kind'].startswith('rgmf_'):
            source_valid=source_valid if source_valid is not None else getattr(self,'_training_source_valid',None)
            if source_valid is None:raise ValueError('Explicit raw-value source mask required; ages and coverage are not availability')
        return super().forward(x,observed,elapsed,source_valid)

    @staticmethod
    def load(path):
        import torch
        from .neural import require_cuda
        require_cuda();state=torch.load(path,map_location='cuda',weights_only=False)
        config=dict(state['config']);mask_id=config.pop('raw_source_mask_id',None)
        adapter=ExplicitSourceCUDA(**config)
        if mask_id:adapter.config['raw_source_mask_id']=mask_id
        adapter.features=state['features'];adapter.model=adapter.build(adapter.features).cuda()
        adapter.model.load_state_dict(state['model']);adapter.model.eval()
        adapter.losses=state['losses'];adapter.validation_losses=state['validation_losses'];adapter.initialization_id=state.get('initialization_id')
        return adapter
