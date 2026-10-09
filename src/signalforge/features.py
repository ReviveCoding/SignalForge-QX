"""Train-only preprocessing and interval-purged chronological partitioning."""
import numpy as np
import pandas as pd
from .pit import mature_training_rows
from .runtime import digest


class TrainTransform:
    def fit(self, x, dates, cutoff):
        x = np.asarray(x, dtype=float)
        dates = pd.to_datetime(dates, utc=True)
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None or len(dates) != len(x) or x.ndim != 2:
            raise ValueError('Invalid training cutoff/shape')
        if len(x) == 0 or (dates > cutoff).any() or np.isinf(x).any():
            raise ValueError('Empty, infinite, or future fitting data')
        # All-missing columns use a deterministic zero; observed masks retain absence.
        self.center = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in x.T])
        filled = np.where(np.isfinite(x), x, self.center)
        self.scale = np.maximum(filled.std(axis=0), 1e-8)
        self.fit_cutoff = cutoff.isoformat()
        self.fit_rows_hash = digest({'dates': dates.astype(str).tolist(),
                                     'values': np.where(np.isfinite(x),x,0).tolist(),'mask':np.isfinite(x).tolist()})
        self.version = 'train-median-standard-v1'
        return self

    def transform(self, x, include_mask=True):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.center) or np.isinf(x).any():
            raise ValueError('Transform dimension/value mismatch')
        observed = np.isfinite(x)
        out = (np.where(observed,x,self.center)-self.center)/self.scale
        return np.concatenate([out,observed.astype(float)],axis=1) if include_mask else out

    def manifest(self):
        return {'center':self.center.tolist(),'scale':self.scale.tolist(),'fit_cutoff':self.fit_cutoff,
                'fit_rows_hash':self.fit_rows_hash,'version':self.version}


def shared_target_scale(y, dates, cutoff):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Invalid target scale sample')
    if pd.Timestamp(cutoff).tzinfo is None or (pd.to_datetime(dates,utc=True)>pd.Timestamp(cutoff)).any():
        raise ValueError('Future target normalizer fit')
    scale = max(float(y.std()), 1e-8)
    return scale, digest({'y':y.tolist(),'dates':pd.to_datetime(dates,utc=True).astype(str).tolist(),'cutoff':str(cutoff)})


def purged_training(frame, cutoff, validation_intervals):
    out = mature_training_rows(frame,cutoff)
    if 'label_start' not in out:
        raise ValueError('Actual target interval starts required')
    start = pd.to_datetime(out.label_start, utc=True)
    end = pd.to_datetime(out.label_end, utc=True)
    if (start >= end).any():
        raise ValueError('Invalid label interval')
    keep = np.ones(len(out), dtype=bool)
    for a,b in validation_intervals:
        a,b = pd.Timestamp(a),pd.Timestamp(b)
        if a.tzinfo is None or b.tzinfo is None or a>=b:
            raise ValueError('Invalid validation interval')
        keep &= ~((start < b) & (end > a)).to_numpy()
    return out.loc[keep].copy()


def dependency_closure(nodes, missing_sources):
    invalid = set(missing_sources)
    changed = True
    while changed:
        previous = len(invalid)
        invalid.update(k for k, parents in nodes.items() if set(parents) & invalid)
        changed = len(invalid) != previous
    return invalid
