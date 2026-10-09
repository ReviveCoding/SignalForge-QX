"""Frozen seed ensembles and target unit inversion, with no final selection API."""
import numpy as np
from .runtime import digest


class FrozenEnsemble:
    def __init__(self,member_ids,weights=None):
        self.member_ids=list(member_ids)
        if not self.member_ids or len(set(self.member_ids))!=len(self.member_ids):
            raise ValueError('Unique frozen members required')
        self.weights=np.ones(len(self.member_ids))/len(self.member_ids) if weights is None else np.asarray(weights,dtype=float)
        if self.weights.shape!=(len(self.member_ids),) or not np.isfinite(self.weights).all() or (self.weights<0).any() or not np.isclose(self.weights.sum(),1):
            raise ValueError('Nonnegative normalized frozen weights required')
        self.id=digest(self.manifest())

    def manifest(self):return {'member_ids':self.member_ids,'weights':self.weights.tolist(),'selection':'frozen_before_calibration'}

    def predict(self,predictions):
        if set(predictions)!=set(self.member_ids):raise ValueError('Ensemble member mismatch')
        means=np.stack([predictions[k][0] for k in self.member_ids]);qs=np.stack([predictions[k][1] for k in self.member_ids])
        if not np.isfinite(means).all() or not np.isfinite(qs).all() or (np.diff(qs,axis=-1)<0).any():
            raise ValueError('Finite ordered member predictions required')
        return np.einsum('s,sn->n',self.weights,means),np.einsum('s,snq->nq',self.weights,qs)


class TargetUnits:
    def __init__(self,center,scale,unit):
        if not np.isfinite(center) or not np.isfinite(scale) or scale<=0 or not unit:raise ValueError('Valid unit transform required')
        self.center,self.scale,self.unit=center,scale,unit

    def inverse(self,mean,quantiles):
        return np.asarray(mean)*self.scale+self.center,np.asarray(quantiles)*self.scale+self.center
