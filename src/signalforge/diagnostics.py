"""Outcome-fixed release interventions rebuilt into a weekly physical feature panel."""
import numpy as np
import pandas as pd

SERIES=['Commercial (Excluding SPR)','Cushing','Distillate Fuel Oil','SPR','Total Motor Gasoline']
FIELDS=[name+'_'+field for name in SERIES for field in ['value','change']]


def release_groups(events):
    releases=[]
    for (raw_hash,reference,available),group in events.groupby(['raw_hash','reference_time','available_at'],sort=False):
        values=group.set_index('field').value
        if values.index.has_duplicates:raise ValueError('Ambiguous source fields')
        if not all(field in values and np.isfinite(values[field]) for field in FIELDS):continue
        if group.available_at.nunique()!=1 or group.reference_time.nunique()!=1:raise ValueError('Inconsistent source release clock')
        releases.append({'raw_hash':raw_hash,'available_at':pd.Timestamp(group.available_at.iloc[0]),
                         'reference_time':pd.Timestamp(group.reference_time.iloc[0]),
                         'values':np.array([values[field] for field in FIELDS],dtype=float)})
    return releases


def weekly_feature(decision,releases):
    decision=pd.Timestamp(decision)
    if decision.tzinfo is None:raise ValueError('Aware origin required')
    known=[r for r in releases if r['available_at']<=decision]
    latest=max(known,key=lambda r:(r['reference_time'],r['available_at'])) if known else None
    physical=latest['values'].copy() if latest else np.full(10,np.nan)
    age=[(decision-latest['available_at']).total_seconds()/86400,(decision-latest['reference_time']).total_seconds()/86400] if latest else [np.nan,np.nan]
    local=decision.tz_convert('America/New_York')
    x=np.concatenate([physical,age,[np.sin(2*np.pi*local.dayofyear/365.25),np.cos(2*np.pi*local.dayofyear/365.25)]])
    return x,latest


def rebuild_physical_features(original,events):
    """Carry genuinely known older releases during outages; preserve every label."""
    out=original.copy(deep=True);releases=release_groups(events);features=[];lineage=[];parents=[]
    for decision in out.decision_time:
        x,latest=weekly_feature(decision,releases);features.append(x.tolist())
        lineage.append(latest['available_at'].isoformat() if latest else pd.Timestamp(decision).isoformat())
        parents.append(latest['raw_hash'] if latest else None)
    out['x']=features;out['max_dependency_available_at']=lineage;out['raw_hash']=parents
    return out
