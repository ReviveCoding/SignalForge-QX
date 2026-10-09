"""Inference from checksum-verified local trained bundles; no fitting or labels."""
import json
from pathlib import Path
import numpy as np
from .runtime import validate_bundle
from .features import TrainTransform
from .models import Statistical,CudaTree,NeuralCUDA
from .ensemble import FrozenEnsemble


def infer_bundle(directory,raw_x,assets=None,context_eligible=None,context_decision_times=None):
    directory=Path(directory)
    validate_bundle(directory)
    metric=json.loads((directory/'metrics.json').read_text())
    manifest=json.loads((directory/'transform.json').read_text())
    transform=TrainTransform();transform.center=np.asarray(manifest['center']);transform.scale=np.asarray(manifest['scale'])
    if not np.isfinite(transform.center).all() or not np.isfinite(transform.scale).all() or (transform.scale<=0).any():
        raise ValueError('Invalid frozen transform')
    x=np.asarray(raw_x,dtype=float)
    if x.ndim!=3 or not len(x) or x.shape[-1]!=len(transform.center):raise ValueError('Frozen context feature shape required')
    observed=np.isfinite(x)
    if metric['family']=='rgmf_residual_gru' and metric.get('bundle_schema')!='multi_asset_residual_v1':
        from .residual import ResidualRGMFCUDA
        return ResidualRGMFCUDA.load(directory/'model.bin').predict(x,observed)
    transformed=transform.transform(x.reshape(-1,x.shape[-1])).reshape(len(x),x.shape[1],-1)
    if metric.get('bundle_schema') in {'multi_asset_normalized_v1','multi_asset_neural_v1','multi_asset_residual_v1'}:
        if assets is None or len(assets)!=len(x) or context_eligible is None:raise ValueError('Frozen multi-asset grid and context eligibility required')
        eligible=np.asarray(context_eligible)
        if eligible.dtype!=bool or eligible.shape!=(len(x),):raise ValueError('Typed context eligibility required')
        scales=json.loads((directory/'asset_scales.json').read_text())
        if not set(assets)<=set(scales):raise ValueError('Asset absent from frozen normalizer')
        scale=np.asarray([scales[asset] for asset in assets],dtype=float)
        if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid per-asset training scales')
        family=metric['family'];flat=transformed.reshape(len(x),-1)
        if metric['bundle_schema']=='multi_asset_residual_v1':
            from .track_residual import TrackResidualCUDA
            mean,q=TrackResidualCUDA.load(directory/'model.bin').predict(x)
        elif metric['bundle_schema']=='multi_asset_neural_v1':
            from .track_neural import ExplicitSourceCUDA,raw_source_validity,elapsed_from_context_clocks
            model=ExplicitSourceCUDA.load(directory/'model.bin');masks=np.concatenate([observed,np.ones_like(observed)],axis=-1)
            valid=raw_source_validity(x,metric['raw_source_validity_columns']) if family.startswith('rgmf_') else None
            elapsed=np.zeros_like(transformed)
            if family=='gru_d':
                if metric.get('elapsed_contract')!='actual_context_days_v1' or context_decision_times is None:
                    raise ValueError('Frozen GRU-D actual context-clock contract required')
                elapsed=elapsed_from_context_clocks(x,context_decision_times)
            mean,q=model.predict(transformed,observed=masks,elapsed=elapsed,source_valid=valid)
        else:
            model=Statistical.load(directory/'model.bin') if family in {'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'} else CudaTree.load(directory/'model.bin')
            mean,q=model.predict(flat)
        fallback=Statistical.load(directory/'fallback.bin')
        fm,fq=fallback.predict(flat[~eligible]);mean[~eligible]=fm;q[~eligible]=fq
        return mean*scale,q*scale[:,None]
    if metric.get('feature_variant'):transformed[...,metric['feature_variant']['removed_transformed_columns']]=0
    family=metric['family']
    if family in {'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}:
        model=Statistical.load(directory/'model.bin');return model.predict(transformed.reshape(len(x),-1))
    if family in {'lightgbm','xgboost','lgb_no_age','lgb_no_coverage'}:
        model=CudaTree.load(directory/'model.bin');return model.predict(transformed.reshape(len(x),-1))
    if family.startswith('concat_capacity_'):
        from .capacity import CapacityMatchedCUDA
        return CapacityMatchedCUDA.load(directory/'model.bin').predict(transformed)
    model=NeuralCUDA.load(directory/'model.bin')
    masks=np.concatenate([observed,observed],axis=-1)
    source_valid=None
    if family.startswith('rgmf_'):
        registered=metric.get('raw_source_validity_columns')
        if not registered:raise ValueError('Frozen RGMF raw source validity registration required')
        source_valid=np.stack([observed[:,-1,columns].any(1) for columns in registered],1)
    return model.predict(transformed,observed=masks,elapsed=np.zeros_like(transformed),source_valid=source_valid)


def infer_candidates(candidates,raw_x,runtime,assets=None,context_eligible=None,context_decision_times=None):
    predictions={}
    for candidate in candidates:
        members={}
        for member in candidate['members']:
            directory=(runtime/member['relative_bundle']).resolve()
            if not directory.is_relative_to(runtime.resolve()):raise PermissionError('Model bundle outside runtime')
            receipt=validate_bundle(directory)
            from .runtime import digest
            if digest(receipt)!=member['receipt_id']:raise ValueError('Frozen model receipt changed')
            members[member['member_id']]=infer_bundle(directory,raw_x,assets,context_eligible,context_decision_times)
        ensemble=FrozenEnsemble([m['member_id'] for m in candidate['members']],candidate['weights'])
        predictions[candidate['candidate_id']]=ensemble.predict(members)
    return predictions
