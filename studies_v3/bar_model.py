"""Small zero-head BAR-GRU extension; exclusively post-result exploratory."""
import numpy as np
import torch
from torch import nn
from signalforge.v33_corrections import BaselineAnchoredResidual

class BAR(nn.Module):
    def __init__(self,dimensions,width=16,bound=.15):
        super().__init__();self.dimensions=list(dimensions);self.width=width;self.bound=bound
        self.encoders=nn.ModuleList([nn.GRU(d,width,batch_first=True) for d in dimensions])
        self.gates=nn.ModuleList([nn.Linear(width,1) for _ in dimensions])
        for gate in self.gates:nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias)
        self.residual=BaselineAnchoredResidual([width]*len(dimensions),bound=bound)
    def forward(self,base,sources,valid,ood):
        states=[];weights=[]
        for i,(encoder,gate,x) in enumerate(zip(self.encoders,self.gates,sources)):
            active=valid[:,i]&~ood[:,i]
            clean=torch.where(active[:,None,None],x,torch.zeros_like(x))
            if not torch.isfinite(clean).all():raise ValueError('Nonfinite active BAR inputs')
            state=encoder(clean)[0][:,-1];states.append(state)
            weights.append(torch.sigmoid(gate(state)).squeeze(1)/len(self.encoders))
        _,q=self.residual(base[:,2],base,states,torch.stack(weights,1),valid,ood)
        return q
    def specification(self):return {'dimensions':self.dimensions,'width':self.width,'bound':self.bound}

def source_ood(transformed,raw_columns):
    ood=np.asarray(transformed['numeric_ood'],dtype=bool)
    return np.stack([ood[...,columns].any(axis=(1,2)) for columns in raw_columns],axis=1)

def source_tensors(values,groups,device='cuda'):
    x=torch.as_tensor(values,dtype=torch.float32,device=device)
    return [x[...,group] for group in groups]

def native_predict(model,values,groups,base_native,scale,valid,ood):
    model.eval();fallback=~(np.asarray(valid)&~np.asarray(ood)).any(axis=1)
    with torch.no_grad():
        q=model(torch.as_tensor(base_native/scale[:,None],dtype=torch.float64,device='cuda'),source_tensors(values,groups),torch.as_tensor(valid,device='cuda'),torch.as_tensor(ood,device='cuda')).cpu().numpy()*scale[:,None]
    q[fallback]=base_native[fallback] # predeclared exact native-unit fallback, not clipping.
    if not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():raise ValueError('Invalid BAR quantiles')
    return q,fallback
