"""Apply frozen ensemble calibration exactly once, preserving raw predictions."""
import numpy as np
import pandas as pd
from .runtime import digest
from .ensemble import FrozenEnsemble
from .models import QUANTILES


def candidate_ensemble_id(candidate):
    members=candidate['members'];ensemble=FrozenEnsemble([m['member_id'] for m in members],candidate['weights'])
    if any(not m.get('receipt_id') or not m.get('relative_bundle') for m in members):raise ValueError('Frozen member receipt identities required')
    return digest({'track':candidate['track'],'candidate_id':candidate['candidate_id'],'members':members,'ensemble':ensemble.manifest()})


def validate_frozen_postprocessing(candidates,ensembles,calibration,minimum_dates=None):
    ids={c['candidate_id'] for c in candidates}
    if len(ids)!=len(candidates):raise ValueError('Unique frozen candidates required')
    minimum_dates=minimum_dates or {'0.05':52,'0.1':39,'0.5':20,'0.9':39,'0.95':52}
    if set(ensembles.get('candidate_ensemble_ids',{}))!=ids or set(calibration.get('candidates',{}))!=ids:
        raise PermissionError('Complete frozen ensemble/calibration family required')
    for candidate in candidates:
        identity=candidate_ensemble_id(candidate);record=calibration['candidates'][candidate['candidate_id']]
        if ensembles['candidate_ensemble_ids'][candidate['candidate_id']]!=identity or record.get('ensemble_id')!=identity:
            raise PermissionError('Calibrator does not belong to frozen model receipts/weights')
        corrections=np.asarray(record.get('corrections'),dtype=float)
        if corrections.shape!=(5,) or not np.isfinite(corrections).all() or record.get('quantiles')!=QUANTILES.tolist():
            raise ValueError('Frozen quantile calibration schema invalid')
        window=record.get('fit_window',{})
        times=[pd.Timestamp(window.get(k)) for k in ['start','end','max_label_available_at']]
        if any(t.tzinfo is None for t in times) or not (pd.Timestamp('2023-07-01T00:00Z')<=times[0]<=times[2]<=times[1]<pd.Timestamp('2024-01-01T00:00Z')):
            raise PermissionError('Frozen calibration window/maturity scope invalid')
        if not record.get('fit_data_id') or set(record.get('support',{}))!={str(float(q)) for q in QUANTILES}:
            raise PermissionError('Calibration fit identity and per-quantile support required')
        for index,q in enumerate(QUANTILES):
            support=record['support'][str(float(q))]
            count=support.get('n_dates')
            if not isinstance(count,int) or isinstance(count,bool) or count<0:raise PermissionError('Distinct calibration date count required')
            if support.get('status')=='IDENTITY_INSUFFICIENT_SUPPORT':
                if corrections[index]!=0:raise PermissionError('Insufficient calibration support must use identity')
            elif support.get('status')=='CORRECTION':
                if count<minimum_dates[str(float(q))]:raise PermissionError('Calibration correction lacks required distinct-date support')
            else:raise PermissionError('Explicit frozen support status required')
    return True


def apply_frozen_postprocessing(candidate,mean,quantiles,calibration):
    record=calibration['candidates'][candidate['candidate_id']]
    if record['ensemble_id']!=candidate_ensemble_id(candidate):raise PermissionError('Frozen ensemble mismatch')
    mean=np.asarray(mean);raw=np.asarray(quantiles)
    before=raw+np.asarray(record['corrections'])
    return {'mean':mean,'raw_quantiles':raw,'before_rearrangement':before,'quantiles':np.sort(before,axis=1),
            'raw_crossing_rows':int((np.diff(raw,axis=1)<0).any(1).sum()),
            'calibration_crossing_rows':int((np.diff(before,axis=1)<0).any(1).sum()),'ensemble_id':record['ensemble_id']}
