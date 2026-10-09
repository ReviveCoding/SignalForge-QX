"""Date-weighted statistical adapters for ragged multi-asset training panels."""
import numpy as np
import pandas as pd
from .models import Statistical,QUANTILES


def weighted_quantiles(values,weights):
    values=np.asarray(values,dtype=float);weights=np.asarray(weights,dtype=float)
    if values.shape!=weights.shape or values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Finite aligned positive-mass quantile weights required')
    keep=weights>0;values=values[keep];weights=weights[keep];order=np.argsort(values,kind='stable')
    cumulative=np.cumsum(weights[order])/weights.sum()
    return values[order][np.searchsorted(cumulative,QUANTILES,side='left')]


def calendar_recency(dates):
    if any(pd.Timestamp(d).tzinfo is None for d in dates):raise ValueError('Aware recency dates required')
    dates=pd.DatetimeIndex(pd.to_datetime(dates,utc=True,format='mixed'))
    days=np.asarray([d.date().toordinal() for d in dates])
    return .97**((days.max()-days)/7.)


class DateWeightedStatistical(Statistical):
    def fit(self,x,y,weights=None,dates=None):
        y=np.asarray(y);weights=np.ones(len(y)) if weights is None else np.asarray(weights)
        if self.family=='ewma':
            if dates is None or len(dates)!=len(y):raise ValueError('EWMA requires aligned calendar dates')
            self.family='historical'
            try:super().fit(x,y,weights=weights*calendar_recency(dates))
            finally:self.family='ewma'
            self.q=weighted_quantiles(y,weights*calendar_recency(dates))
        else:
            super().fit(x,y,weights=weights)
            if self.family=='historical':self.q=weighted_quantiles(y,weights)
            if self.family in {'ridge','mixed_frequency_shrinkage'}:
                self.residual_q=weighted_quantiles(y-self.mean_model.predict(x),weights)
        self.quantile_method='weighted_inverse_empirical_cdf_v1'
        return self
