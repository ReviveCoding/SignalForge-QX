"""Shared outcome-blind grids, release-aware feature panels and track folds."""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from .data import validate_events,valid_mapping
from .pit import asof_snapshot
from .features import purged_training
from .runtime import digest


@dataclass(frozen=True)
class FeatureSpec:
    name:str
    source:str
    field:str
    entity:str
    unit:str
    mapping:bool=False
    percentile_window:int=0


def fixed_grid(decisions,universe_card,reserved_start='2024-01-01T00:00Z'):
    if universe_card.get('outcome_blind') is not True or not universe_card.get('frozen_before_outcomes'):
        raise PermissionError('Documented outcome-blind candidate universe required')
    assets=list(universe_card['assets'])
    if not assets or len(assets)!=len(set(assets)):raise ValueError('Unique candidate assets required')
    if any(pd.Timestamp(d).tzinfo is None for d in decisions):raise ValueError('Aware decision origins required')
    dates=pd.to_datetime(decisions,utc=True)
    if len(dates)!=len(dates.unique()) or not dates.is_monotonic_increasing:
        raise ValueError('Unique chronological decision origins required')
    if len(dates) and dates.max()>=pd.Timestamp(reserved_start):raise PermissionError('Development grid cannot access reserved dates')
    grid=pd.DataFrame([{'decision_time':d.isoformat(),'asset':asset,'eligible':True} for d in dates for asset in assets])
    grid.attrs['eligibility_id']=digest({'dates':dates.astype(str).tolist(),'universe_card':universe_card})
    return grid


def build_features(grid,events,specs,information_sources,mappings=None,strict=False,evidence_registry=None,allow_invalid_control=False):
    """All model families consume this same value/age/mask/coverage vector."""
    validate_events(events,evidence_registry)
    if events.pit_tier.eq('C').any() and not allow_invalid_control:raise PermissionError('Tier C is restricted to explicitly invalid development controls')
    specs=[s for s in specs if s.source in information_sources]
    if not specs or len({s.name for s in specs})!=len(specs):raise ValueError('Unique registered feature names required')
    active_events=events[events.source.isin(information_sources)].copy()
    out=grid.copy();vectors=[];lineages=[];names=[];groups={};value_groups={};meta=[]
    for spec in specs:
        position=len(names)
        names.extend([spec.name,spec.name+'_release_age',spec.name+'_reference_age',spec.name+'_observed',spec.name+'_coverage'])
        if spec.percentile_window:names.append(spec.name+'_past_percentile')
        groups.setdefault(spec.source,[]).extend(range(position,len(names)))
        value_groups.setdefault(spec.source,[]).append(position)
        if spec.source!='market':meta.extend(range(position+1,position+5))
    season=list(range(len(names),len(names)+2))
    names+=['season_sin','season_cos']
    snapshots={}
    for row in grid.itertuples():
        decision=pd.Timestamp(row.decision_time)
        if decision.tzinfo is None:raise ValueError('Aware origin required')
        if row.decision_time not in snapshots:snapshots[row.decision_time]=asof_snapshot(active_events,decision,strict)
        snapshot=snapshots[row.decision_time]
        vector=[];parents=[];max_known=None
        for spec in specs:
            entity=row.asset if spec.entity=='ASSET' else spec.entity
            if spec.mapping:
                if mappings is None:raise ValueError('As-of mappings required')
                entity=valid_mapping(mappings,entity,decision,decision)
            selected=snapshot[snapshot.entity.eq(entity)&snapshot.source.eq(spec.source)&snapshot.field.eq(spec.field)]
            if len(selected)>1:raise ValueError('Ambiguous feature join')
            if len(selected):
                event=selected.iloc[0]
                if event.unit!=spec.unit:raise ValueError('Registered feature unit conflict')
                value=float(event.value);observed=np.isfinite(value);known=pd.Timestamp(event.available_at)
                vector.extend([value,float(event.release_age_days),float(event.reference_age_days),float(observed),float(observed)])
                parents.append({'feature':spec.name,'raw_hash':event.raw_hash,'reference_time':pd.Timestamp(event.reference_time).isoformat(),'available_at':known.isoformat(),'unit':event.unit})
                max_known=known if max_known is None else max(max_known,known)
                if spec.percentile_window:
                    history=active_events[active_events.entity.eq(entity)&active_events.source.eq(spec.source)&active_events.field.eq(spec.field)].copy()
                    history['available_at']=pd.to_datetime(history.available_at,utc=True);history['reference_time']=pd.to_datetime(history.reference_time,utc=True)
                    history=history[history.available_at.le(decision)]
                    if strict:history=history[history.pit_tier.eq('A')]
                    history=history.sort_values(['reference_time','available_at']).drop_duplicates('reference_time',keep='last').sort_values('reference_time').tail(spec.percentile_window)
                    values=pd.to_numeric(history.value).dropna().to_numpy()
                    vector.append(float(np.mean(values<=value)) if len(values) and observed else np.nan)
            else:
                vector.extend([np.nan,np.nan,np.nan,0.,0.])
                if spec.percentile_window:vector.append(np.nan)
        local=decision.tz_convert('America/New_York');vector.extend([np.sin(2*np.pi*local.dayofyear/365.25),np.cos(2*np.pi*local.dayofyear/365.25)])
        vectors.append(vector);lineages.append({'parents':parents,'max_dependency_available_at':(max_known or decision).isoformat()})
    out['x']=vectors;out['feature_lineage']=lineages
    out['max_dependency_available_at']=[r['max_dependency_available_at'] for r in lineages]
    out.attrs.update(grid.attrs);out.attrs['feature_names']=names
    out.attrs['information_id']=digest({'specs':[asdict(s) for s in specs],'sources':sorted(information_sources),'strict':strict})
    out.attrs['promotion_eligible']=not events.pit_tier.eq('C').any()
    out.attrs['pit_tiers']=sorted(events.pit_tier.unique().tolist())
    sources=sorted(set(groups)-{'market'})
    out.attrs['neural_contract']={'base_raw_columns':groups.get('market',[])+season,
        'source_raw_columns':[groups[source] for source in sources],
        'source_value_columns':[value_groups[source] for source in sources],
        'meta_raw_columns':meta or season,'source_names':sources}
    return out


def matched_grid(*panels):
    """Fail on missing dates/assets instead of intersecting away hard origins."""
    if not panels:raise ValueError('Comparison panels required')
    keys=['decision_time','asset'];base=panels[0][keys].sort_values(keys).reset_index(drop=True)
    if base.duplicated(keys).any():raise ValueError('Duplicate eligible grid keys')
    for panel in panels[1:]:
        other=panel[keys].sort_values(keys).reset_index(drop=True)
        if not other.equals(base):raise ValueError('Matched comparison grid differs')
    return digest(base.to_dict('records'))


def track_folds(panel,study,track):
    contract=study['splits']['track_folds'][track];folds={};blocked=[]
    minimum=contract.get('min_train_decision_dates',contract.get('min_train_release_events'))
    for year in contract['outer_years']:
        dates=pd.to_datetime(panel.decision_time,utc=True)
        test=panel[dates.ge(f'{year}-01-01T00:00Z')&dates.lt(f'{year+1}-01-01T00:00Z')]
        labeled=panel[np.isfinite(pd.to_numeric(panel.y))&panel[['label_start','label_end','label_available_at']].notna().all(axis=1)]
        intervals=test.dropna(subset=['label_start','label_end'])
        train=purged_training(labeled,pd.Timestamp(f'{year}-01-01T00:00Z')-pd.Timedelta(nanoseconds=1),list(zip(intervals.label_start,intervals.label_end)))
        train=train[np.isfinite(train.y)]
        if train.decision_time.nunique()<minimum or test.empty:
            blocked.append({'track':track,'outer_year':year,'state':'BLOCKED_DATA','reason':'Distinct chronological training/outer coverage insufficient'});continue
        valid_dates=sorted(train.decision_time.unique())[-26:];valid=train[train.decision_time.isin(valid_dates)]
        cutoff=pd.Timestamp(valid_dates[0])-pd.Timedelta(nanoseconds=1)
        inner=purged_training(train,cutoff,list(zip(valid.label_start,valid.label_end)))
        if valid.decision_time.nunique()<contract.get('inner_validation_min_decision_dates',13) or inner.empty:
            blocked.append({'track':track,'outer_year':year,'state':'BLOCKED_DATA','reason':'Inner chronological history insufficient'});continue
        folds[year]={'train':train,'test':test,'inner':inner,'valid':valid,'inner_cutoff':cutoff,'outer_cutoff':pd.Timestamp(f'{year}-01-01T00:00Z')-pd.Timedelta(nanoseconds=1)}
    return folds,blocked


def build_contexts(panel,lookback=26):
    """Retain origins and distinguish padding, missing observations and eligibility."""
    if lookback<1 or panel[['asset','decision_time']].duplicated().any():raise ValueError('Unique positive context contract required')
    out=panel.sort_values(['asset','decision_time']).reset_index(drop=True).copy()
    if out.empty:raise ValueError('Nonempty feature panel required')
    dimension=len(out.x.iloc[0]);contexts=[];padding=[];observed=[];eligibility=[];context_lineage=[]
    for asset,group in out.groupby('asset',sort=False):
        group=group.sort_values('decision_time')
        for position,index in enumerate(group.index):
            history=group.iloc[max(0,position-lookback+1):position+1]
            if any(len(v)!=dimension for v in history.x):raise ValueError('Context feature schema changed')
            decisions=pd.to_datetime(history.decision_time,utc=True);cutoff=pd.Timestamp(out.loc[index,'decision_time'])
            known=pd.to_datetime(history.max_dependency_available_at,utc=True)
            if (known>decisions).any():raise ValueError('Context contains a future feature dependency')
            padded=lookback-len(history);matrix=np.full((lookback,dimension),np.nan)
            matrix[padded:]=np.asarray(history.x.tolist(),dtype=float)
            pad=np.arange(lookback)<padded
            # Every family can distinguish padding from an ordinary missing value.
            contexts.append(np.column_stack([matrix,pad.astype(float)]));padding.append(pad);observed.append(np.isfinite(matrix))
            context_lineage.append([{'decision_time':str(r.decision_time),'feature_lineage':r.feature_lineage,
                'max_dependency_available_at':str(r.max_dependency_available_at)} for r in history.itertuples()])
            contiguous=not (decisions.diff().dropna()>pd.Timedelta(days=10)).any()
            eligibility.append(bool(out.loc[index,'eligible']) and padded==0 and contiguous)
    out['sequence_index']=np.arange(len(out));out['context_eligible']=eligibility
    out['context_lineage']=context_lineage
    out['context_status']=['ELIGIBLE' if value else 'INSUFFICIENT_CONTEXT_OR_GAP' for value in eligibility]
    out.attrs.update(panel.attrs);out.attrs['feature_names']=panel.attrs.get('feature_names',[])+['context_padding_indicator']
    if 'neural_contract' in out.attrs:
        contract={k:list(v) for k,v in out.attrs['neural_contract'].items()}
        contract['base_raw_columns']=contract['base_raw_columns']+[dimension]
        out.attrs['neural_contract']=contract
    return out,np.asarray(contexts),{'padding':np.asarray(padding),'observed':np.asarray(observed),'eligible':np.asarray(eligibility)}
