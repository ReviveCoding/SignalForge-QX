"""Calibration is fitted state tied to a frozen ensemble and mature window."""
import numpy as np
import pandas as pd
from .runtime import digest


class Calibrator:
    def fit(self, y, predictions, dates, label_available, ensemble_id, start, end, minimum_dates):
        start,end = pd.Timestamp(start),pd.Timestamp(end)
        dates = pd.to_datetime(dates,utc=True)
        mature = pd.to_datetime(label_available,utc=True)
        if start.tzinfo is None or end.tzinfo is None or not ensemble_id or start>end:
            raise ValueError('Invalid calibration contract')
        y,predictions = np.asarray(y),np.asarray(predictions)
        qs = np.array(sorted(float(q) for q in minimum_dates))
        if predictions.shape != (len(y),len(qs)) or len(dates)!=len(y) or len(mature)!=len(y):
            raise ValueError('Calibration shape mismatch')
        if ((dates<start)|(dates>end)|(mature>end)).any() or not np.isfinite(predictions).all() or not np.isfinite(y).all():
            raise ValueError('Calibration scope or finite-data violation')
        n = len(dates.unique())
        self.corrections = np.array([np.quantile(y-predictions[:,i],q) if n>=minimum_dates[str(q)] else 0
                                     for i,q in enumerate(qs)])
        self.support = {str(q):{'n_dates':n,'status':'CORRECTION' if n>=minimum_dates[str(q)] else 'IDENTITY_INSUFFICIENT_SUPPORT'} for q in qs}
        self.ensemble_id = ensemble_id
        self.quantiles=qs
        self.fit_window={'start':start.isoformat(),'end':end.isoformat(),'max_label_available_at':mature.max().isoformat()}
        self.fit_data_id=digest({'y':y.tolist(),'predictions':predictions.tolist(),'dates':dates.astype(str).tolist(),
                                 'label_available':mature.astype(str).tolist(),'window':self.fit_window})
        self.id = digest({'ensemble':ensemble_id,'support':self.support,'corrections':self.corrections.tolist(),'fit_data_id':self.fit_data_id})
        return self

    def manifest(self):
        return {'ensemble_id':self.ensemble_id,'calibration_id':self.id,'corrections':self.corrections.tolist(),
                'quantiles':self.quantiles.tolist(),'support':self.support,'fit_window':self.fit_window,
                'fit_data_id':self.fit_data_id,'method':'additive_residual_quantile_then_registered_rearrangement'}

    def transform(self, predictions, ensemble_id):
        if ensemble_id != self.ensemble_id:
            raise ValueError('Calibrator belongs to a different ensemble')
        raw = np.asarray(predictions)+self.corrections
        return {'before_rearrangement':raw,'processed':np.sort(raw,axis=1),
                'crossing_rows':int((np.diff(raw,axis=1)<0).any(axis=1).sum())}
