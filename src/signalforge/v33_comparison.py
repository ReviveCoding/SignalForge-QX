"""Strict paired diagnostic score oracle; requires real v3.3 predictions to report gains."""
import numpy as np
import pandas as pd
from .metrics import quantile_loss

Q=np.array([.05,.1,.5,.9,.95])
KEYS=['decision_time','asset','seed']
COLS=['q0.05','q0.1','q0.5','q0.9','q0.95']
def paired_comparison(old,new,old_meta,new_meta):
    for key in ['fold_id','source_id','normalizer_id','target_contract_id']:
        if not old_meta.get(key) or old_meta[key]!=new_meta.get(key):
            raise PermissionError('Paired immutable '+key+' mismatch')
    frames=[]
    for frame in [old,new]:
        f=frame.copy()
        if not set(KEYS+COLS+['target','scale'])<=set(f):
            raise ValueError('Complete paired predictions and targets required')
        f['decision_time']=pd.to_datetime(f.decision_time,utc=True)
        if (f.decision_time>=pd.Timestamp('2024-01-01T00:00Z')).any():
            raise PermissionError('Reserved comparison forbidden')
        if f.empty or f.duplicated(KEYS).any():raise ValueError('Unique nonempty grid')
        f=f.sort_values(KEYS).reset_index(drop=True); frames.append(f)
    a,b=frames
    if not a[KEYS].equals(b[KEYS]):raise PermissionError('No dropping unmatched rows')
    if not np.array_equal(a.target,b.target) or not np.array_equal(a.scale,b.scale):
        raise PermissionError('Shared targets and scale must match exactly')
    if not set(a.seed)=={11,37,71}:raise PermissionError('All three fixed seeds required')
    grid=a.groupby('decision_time').apply(lambda g:set(zip(g.asset,g.seed)),include_groups=False)
    if not all(g==grid.iloc[0] for g in grid):raise PermissionError('No collapsing variable date/seed cohorts')
    losses=[]
    for f in frames:
        q=f[COLS].to_numpy(float); y=f.target.to_numpy(float); scale=f.scale.to_numpy(float)
        if not np.isfinite(q).all() or not np.isfinite(y).all() or not np.isfinite(scale).all() or (scale<=0).any() or (np.diff(q,axis=1)<0).any():
            raise ValueError('Finite ordered predictions and positive shared scales')
        loss=quantile_loss(y,q,Q,scale)
        absolute=quantile_loss(y,q,Q,np.ones(len(y)))
        per_date=pd.DataFrame({'decision_time':f.decision_time,'seed':f.seed,'pinball':loss,'absolute':absolute})
        # Asset average inside seed/date; seed average inside date; time average last.
        per_seed=per_date.groupby(['decision_time','seed'])[['pinball','absolute']].mean()
        dates=per_seed.groupby('decision_time').mean()
        losses.append({'normalized_pinball':float(dates.pinball.mean()),'absolute_pinball':float(dates.absolute.mean()),
            'p99_abs_quantile_forecast':float(np.quantile(np.abs(q).max(axis=1),.99)),
            'forecast_rows_abs_over100':int((np.abs(q).max(axis=1)>100).sum()),
            'per_quantile_coverage':(y[:,None]<=q).mean(axis=0).tolist(),
            'seed_pinball':{str(k):float(v) for k,v in per_seed.groupby('seed').pinball.mean().items()}})
    baseline=losses[0]['normalized_pinball']
    return {'old':losses[0],'new':losses[1],'independent_dates':int(a.decision_time.nunique()),
        'forecast_rows':len(a),'rows_dropped':0,'relative_pinball_gain':1-losses[1]['normalized_pinball']/baseline if baseline>0 else None,
        'scope':'retrospective_diagnostic_only','qualified_for_final':False}
