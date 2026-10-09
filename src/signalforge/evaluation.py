"""Fixed grids, date-level losses, paired blocks and full-family multiplicity."""
import numpy as np
import pandas as pd
from .metrics import quantile_loss, paired_time_block_ci


def score_grid(grid, predictions, fallback, quantiles, scale):
    keys = ['decision_time','asset']
    for frame in [grid,predictions,fallback]:
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate comparison grid key')
    qcols = [f'q{q:g}' for q in quantiles]
    joined = grid.merge(predictions[keys+qcols],on=keys,how='left',validate='one_to_one')
    f = grid[keys].merge(fallback[keys+qcols],on=keys,how='left',validate='one_to_one')
    if not np.isfinite(f[qcols].to_numpy()).all():
        raise ValueError('Registered fallback must cover complete grid')
    failed = ~np.isfinite(joined[qcols].to_numpy()).all(axis=1)
    joined.loc[failed,qcols] = f.loc[failed,qcols].to_numpy()
    loss = quantile_loss(joined.y.to_numpy(),joined[qcols].to_numpy(),quantiles,scale)
    joined['loss'] = loss
    by_date = joined.groupby('decision_time',sort=True).loss.mean()
    return {'score':float(by_date.mean()),'n_unique_dates':len(by_date),'n_rows':len(joined),
            'fallback_count':int(failed.sum()),'native_prediction_coverage':float((~failed).mean()),
            'date_losses':{str(k):float(v) for k,v in by_date.items()}}


def holm(pvalues):
    """Missing registered contrasts retain family size, conservatively treated as p=1."""
    keys = list(pvalues)
    p = np.array([1.0 if pvalues[k] is None else pvalues[k] for k in keys])
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid p values')
    order = np.argsort(p,kind='stable')
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1)
    return {keys[i]:None if pvalues[keys[i]] is None else float(adjusted[j]) for j,i in enumerate(order)}


def compare(left,right,block=8,draws=2000):
    if list(left) != list(right):
        raise ValueError('Unmatched ordered decision grid')
    dates = pd.to_datetime(list(left),utc=True)
    if len(dates)>1 and np.max(np.diff(dates.asi8)) > 10*86400*10**9:
        raise ValueError('Segment missing calendar gaps before block resampling')
    return paired_time_block_ci(np.array(list(left.values()))-np.array(list(right.values())),block,draws)
