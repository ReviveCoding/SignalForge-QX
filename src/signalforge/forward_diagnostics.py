"""Separate future-study admission and target-group exclusion contracts."""
import numpy as np
import pandas as pd
from .features import TrainTransform,shared_target_scale


def planned_fit_counts(plan):
    folds=plan['cohort']['evaluation_blocks'];tracks=len(plan['tracks']);per=plan['hpo']['trials_per_family_fold']+len(plan['seeds'])
    if len(plan['cohort']['assets'])!=len(set(plan['cohort']['assets'])):raise ValueError('Unique physical assets required')
    counts={'matched_primary':2*tracks*folds*per,'retrained_components':len(plan['retrained_components'])*tracks*folds*per,'ssl_paired':2*tracks*folds*per,
            'zero_shot_physical_groups':len(plan['cohort']['assets'])*tracks*2*folds*per}
    counts['total']=sum(counts.values());return counts


def zero_shot_preprocessing(frame,held_out,cutoff):
    """Unknown groups use a pooled training-only scale, never their own labels."""
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff>=pd.Timestamp('2024-01-01T00:00Z'):
        raise PermissionError('Development-only transport preprocessing cutoff')
    train=frame.loc[~frame.asset.isin(held_out)].copy()
    if set(train.asset)&set(held_out):raise ValueError('Held-out group leakage')
    keep=(pd.to_datetime(train.decision_time,utc=True)<=cutoff)&(pd.to_datetime(train.label_available_at,utc=True)<=cutoff)
    train=train.loc[keep]
    if not len(train) or train.asset.nunique()<1:raise ValueError('No independent source group')
    scale,identity=shared_target_scale(train.y.to_numpy(dtype=float),train.decision_time,cutoff)
    transform=TrainTransform().fit(np.array(train.x.tolist(),dtype=float),train.decision_time,cutoff)
    return {'training_groups':sorted(train.asset.unique()),'held_out_groups':sorted(held_out),
            'shared_scale':scale,'normalizer_id':identity,'transform':transform.manifest(),
            'unknown_group_embedding':'none; shared feature encoder only',
            'unknown_group_normalizer':'pooled source-group scale; no held-out target fit'}


def admission(plan,evidence):
    failures=[]
    required=['explicit_separate_run_authorization','current_full_validation','measured_pilot_bound_to_plan',
              'original_source_clock_qualified','operational_freeze_verified','new_cohort_unseen','future_cohort_available']
    for key in required:
        if evidence.get(key) is not True:failures.append(key)
    projected=evidence.get('conservative_gpu_seconds')
    ceiling=plan['budget']['requested_ceiling_gpu_seconds']
    if projected is None or not np.isfinite(projected) or projected<0 or projected>ceiling:
        failures.append('measured_resource_bill_within_separate_authorized_ceiling')
    if evidence.get('use_reserved_data') is not False:failures.append('reserved_data_sealed')
    return {'state':'ADMITTED' if not failures else 'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE',
            'missing_dependencies':failures,'outcome_blind':True,'training_started':False}
