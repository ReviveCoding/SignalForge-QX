"""Typed experiment identities, event-level outage schedules and train-only rules."""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from .features import dependency_closure
from .runtime import digest


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    track: str
    partition: str
    intervention: str
    retrained: bool
    configuration: dict
    promotion_eligible: bool=True

    def validate(self):
        if self.experiment not in {f'E{i:02d}' for i in range(1,11)}:raise ValueError('Unknown experiment')
        if self.partition not in {'development','selection','calibration','reserved'}:raise ValueError('Unknown partition')
        if self.intervention not in {'information','architecture','retrained_ablation','frozen_input','explanation','invalid_control','transport_zero_shot','transport_refit','scale_weighting','ssl','economics','systems'}:
            raise ValueError('Explicit intervention type required')
        if self.intervention=='retrained_ablation' and not self.retrained:raise ValueError('Retrained ablation must refit')
        if self.intervention in {'frozen_input','explanation','transport_zero_shot'} and self.retrained:raise ValueError('Frozen intervention cannot refit')
        if self.intervention=='transport_refit' and not self.retrained:raise ValueError('Refit transport must refit')
        if self.experiment=='E04' and (self.promotion_eligible or self.intervention!='invalid_control'):
            raise PermissionError('Invalid controls cannot be promoted')
        if self.experiment in {'E04','E07','E08','E10'} and self.partition!='development':
            raise PermissionError('Development-only experiment')
        if self.partition=='reserved' and not self.configuration.get('frozen_receipt_id'):
            raise PermissionError('Frozen final registration required')
        return digest(asdict(self))


def apply_release_outage(events,schedule,dependencies):
    """Corrupt public source releases before rebuilding all derived features.

    start/end define aware release-time blocks; delay is calendar days. An
    unavailable event remains explicitly missing. No label values are consulted.
    """
    out=events.copy();invalid=set();trace=[]
    out['available_at']=pd.to_datetime(out.available_at,utc=True)
    for block in schedule:
        source=block['source'];start,end=pd.Timestamp(block['start']),pd.Timestamp(block['end'])
        if start.tzinfo is None or end.tzinfo is None or start>=end:raise ValueError('Aware valid release block required')
        mask=out.source.eq(source)&out.available_at.ge(start)&out.available_at.lt(end)
        if block['kind']=='blackout':out.loc[mask,'value']=np.nan
        elif block['kind']=='delay':
            if block['days']<=0:raise ValueError('Positive delay required')
            out.loc[mask,'available_at']+=pd.Timedelta(days=block['days'])
        else:raise ValueError('Unknown outage intervention')
        invalid.update(dependency_closure(dependencies,[source]))
        trace.append({'source':source,'kind':block['kind'],'affected_release_rows':int(mask.sum()),'schedule':block})
    return out,sorted(invalid),trace


def training_weights(rule,dates,lagged_volatility,cutoff):
    dates=pd.to_datetime(dates,utc=True);cut=pd.Timestamp(cutoff)
    if cut.tzinfo is None or not len(dates) or (dates>cut).any():raise ValueError('Weights require past training rows')
    if rule=='uniform':weights=np.ones(len(dates))
    elif rule=='recency':weights=np.exp(-np.log(2)*(cut-dates).total_seconds().to_numpy()/(365.25*86400))
    elif rule=='lagged_vol_regime_balanced':
        volatility=np.asarray(lagged_volatility)
        if volatility.shape!=(len(dates),) or not np.isfinite(volatility).all() or (volatility<0).any():raise ValueError('Past finite lagged volatility required')
        boundaries=np.quantile(volatility,[1/3,2/3]);groups=np.searchsorted(boundaries,volatility,side='right')
        counts=np.bincount(groups,minlength=3);weights=np.array([1/counts[g] for g in groups])
    else:raise ValueError('Unknown registered weighting rule')
    return weights/weights.mean()


def scale_weight_grid(small_rule_receipt,scales=(100000,500000,2000000),seeds=(11,37,71),folds=(2021,2022)):
    if small_rule_receipt.get('partition')!='inner_validation' or not small_rule_receipt.get('frozen_rule_hash'):
        raise PermissionError('Small-scale weighting rule must be frozen from inner development')
    return [{'experiment':'E07','parameters':scale,'rule':rule,'seed':seed,'fold':fold,
             'small_rule_hash':small_rule_receipt['frozen_rule_hash'],'promotion_eligible':False,
             'exposure_table':'fixed_samples','compute_table':'fixed_elapsed_budget'}
            for scale in scales for rule in ['uniform','recency','lagged_vol_regime_balanced'] for seed in seeds for fold in folds]


def invalid_control(events,kind,partition):
    if partition!='development':raise PermissionError('Leakage control cannot access held-out partitions')
    out=events.copy()
    if kind=='wrong_release':out['available_at']=out['reference_time']
    elif kind=='latest_revision':
        keys=['entity','source','field','reference_time']
        out=out.sort_values('available_at').drop_duplicates(keys,keep='last')
        out['available_at']=out['reference_time']
    else:raise ValueError('Unknown invalid source-clock control')
    out['pit_tier']='C';out['invalid_control']=True;out['promotion_eligible']=False
    return out
