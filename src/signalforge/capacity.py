"""Outcome-independent concatenation controls matched to RGMF parameter budgets."""
import math
from .models import NeuralCUDA


def match_capacity(features,target,kind,target_width,tolerance=.01):
    candidates=[]
    for width in range(4,513,4) if kind=='transformer' else range(4,513):
        if kind=='gru':
            base=6*width**2+(features+7)*width+6
            hidden=max(1,round((target-base)/(width+7)))
            count=base+hidden*(width+7)
            spec={'width':width,'head_width':hidden,'feedforward':None}
        elif kind=='transformer':
            base=4*width**2+(features+16)*width+6
            hidden=max(1,round((target-base)/(2*width+1)))
            count=base+hidden*(2*width+1)
            spec={'width':width,'head_width':None,'feedforward':hidden}
        else:raise ValueError('Capacity controls support GRU/Transformer')
        if hidden>4096 or count>500000 or abs(count-target)/target>tolerance:continue
        candidates.append((abs(width-math.sqrt(2)*target_width),abs(count-target),width,spec,count))
    if not candidates:raise ValueError('No control meets preregistered 1% parameter tolerance')
    _,_,_,spec,count=min(candidates,key=lambda row:row[:3])
    return {**spec,'target_parameters':target,'control_parameters':count,'relative_parameter_gap':abs(count-target)/target,'tolerance':tolerance}


class CapacityMatchedCUDA(NeuralCUDA):
    """Same complete information, six mean/ordered-quantile heads and CUDA optimizer."""
    def __init__(self,kind='gru',width=32,epochs=100,seed=11,lr=.003,capacity_reference=None,**kwargs):
        if kind not in {'gru','transformer'}:raise ValueError('Unsupported capacity-control kind')
        super().__init__(kind,width,epochs,seed,lr,**kwargs)
        self.config['capacity_reference']=capacity_reference or {'base_features':8,'source_features':[20],'meta_features':4,'full_features':28}

    def build(self,features):
        from torch import nn
        from .neural import Encoder,RGMF
        c=self.config;reference=c['capacity_reference']
        if features!=reference['full_features']:raise ValueError('Capacity control registered feature schema changed')
        target=RGMF(reference['base_features'],reference['source_features'],reference['meta_features'],c['width'],c['kind'])
        count=sum(p.numel() for p in target.parameters())
        spec=match_capacity(features,count,c['kind'],c['width']);self.capacity_spec=spec
        model=Encoder(features,spec['width'],c['kind'])
        if c['kind']=='gru':model.output=nn.Sequential(nn.Linear(spec['width'],spec['head_width']),nn.GELU(),nn.Linear(spec['head_width'],6))
        else:model.attention=nn.TransformerEncoder(nn.TransformerEncoderLayer(spec['width'],4,spec['feedforward'],dropout=0,batch_first=True),1)
        actual=sum(p.numel() for p in model.parameters())
        if actual!=spec['control_parameters'] or abs(actual-count)/count>spec['tolerance']:raise ValueError('Analytical/actual parameter count mismatch')
        return model

    @staticmethod
    def load(path):
        import torch
        from .neural import require_cuda
        require_cuda();state=torch.load(path,map_location='cuda',weights_only=False)
        adapter=CapacityMatchedCUDA(**state['config']);adapter.features=state['features']
        adapter.model=adapter.build(adapter.features).cuda();adapter.model.load_state_dict(state['model']);adapter.model.eval()
        adapter.losses=state['losses'];adapter.validation_losses=state['validation_losses'];adapter.initialization_id=state.get('initialization_id')
        return adapter
