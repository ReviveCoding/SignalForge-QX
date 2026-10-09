"""Post-hoc diagnostics of immutable development forecasts; never training."""
import numpy as np
import pandas as pd
from .runtime import digest

QUANTILES=np.array([.05,.1,.5,.9,.95])


def price_mark_index(records):
    frame=pd.DataFrame(records)
    if not {'asset','time','close','raw_hash'}<=set(frame):raise ValueError('Price mark provenance required')
    times=pd.to_datetime(frame.time,utc=True)
    if times.ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved price marks forbidden')
    keys=list(zip(frame.asset,times.astype(str)))
    if len(set(keys))!=len(keys):raise ValueError('Ambiguous original price mark')
    return {key:row for key,row in zip(keys,frame.itertuples())}


def feature_lineage_ids(testing):
    if 'feature_lineage' not in testing:raise ValueError('Original per-feature provenance required')
    return testing.feature_lineage.map(digest).to_numpy()


def forecast_rows(testing,mean,quantiles,scales):
    dates=pd.to_datetime(testing.decision_time,utc=True)
    if dates.ge('2024-01-01T00:00Z').any():
        raise PermissionError('Reserved outcomes forbidden in forensic development audit')
    y=testing.y.to_numpy(dtype=float);mean=np.asarray(mean);q=np.asarray(quantiles);s=np.asarray(scales)
    if q.shape!=(len(y),5) or mean.shape!=y.shape or s.shape!=y.shape:
        raise ValueError('Aligned original forecast grid required')
    if not np.isfinite(q).all() or not np.isfinite(mean).all() or not np.isfinite(s).all() or (s<=0).any():
        raise ValueError('Finite predictions and positive original scales required')
    if (np.diff(q,axis=1)<0).any():raise ValueError('Original quantile crossing')
    out=testing[['decision_time','asset']].copy();out['decision_time']=dates.astype(str)
    out['target']=y;out['mean']=mean;out['scale']=s
    out['scorable']=testing.eligible.to_numpy(dtype=bool)&np.isfinite(y)
    error=y[:,None]-q;pin=np.maximum(error*QUANTILES,error*(QUANTILES-1)).mean(axis=1)
    out['absolute_pinball']=np.where(out.scorable,pin,np.nan)
    out['normalized_pinball']=out.absolute_pinball/s
    out['normalized_mse']=np.where(out.scorable,((y-mean)/s)**2,np.nan)
    out['forecast_abs_max']=np.abs(q).max(axis=1)
    out['forecast_scaled_abs_max']=np.abs(q/s[:,None]).max(axis=1)
    out['interval_width']=q[:,-1]-q[:,0]
    for i,level in enumerate(QUANTILES):out['q'+str(level)]=q[:,i]
    return out


def extrapolation(raw,manifest,names):
    raw=np.asarray(raw,dtype=float);center=np.array(manifest['center']);scale=np.array(manifest['scale'])
    if raw.ndim!=3 or len(names)!=raw.shape[-1] or center.shape!=scale.shape or len(center)!=len(names):
        raise ValueError('Original feature/transform schema required')
    if not np.isfinite(scale).all() or (scale<=0).any() or np.isinf(raw).any():raise ValueError('Invalid immutable transform')
    z=(np.where(np.isfinite(raw),raw,center)-center)/scale
    flat=np.abs(z).reshape(len(z),-1);indices=flat.argmax(axis=1);columns=indices%len(names)
    rows={'transformed_abs_max':flat.max(axis=1),'dominant_feature':[names[i] for i in columns],
          'dominant_raw_value':raw.reshape(len(raw),-1)[np.arange(len(raw)),indices],
          'dominant_training_center':center[columns],'dominant_training_scale':scale[columns]}
    summary=[]
    for i,name in enumerate(names):
        values=raw[:,:,i];seen=values[np.isfinite(values)]
        summary.append({'feature':name,'training_center':float(center[i]),'training_scale':float(scale[i]),
                        'scale_floor':bool(scale[i]==1e-8),'test_observed':int(len(seen)),
                        'test_abs_z_max':float(np.abs(z[:,:,i]).max()),
                        'test_min':float(seen.min()) if len(seen) else None,'test_max':float(seen.max()) if len(seen) else None})
    return rows,summary


def date_score(rows,column='normalized_pinball'):
    return float(rows.groupby('decision_time')[column].mean().mean())


def relative_gain(baseline,candidate):
    if not np.isfinite(baseline) or not np.isfinite(candidate):raise ValueError('Finite comparison required')
    return (baseline-candidate)/baseline if baseline>0 else None
