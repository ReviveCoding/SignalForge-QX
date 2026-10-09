"""Strictly pre-outer OOF calibration; unsupported quantiles remain exact identity."""
import numpy as np,pandas as pd
Q=np.array([.05,.1,.5,.9,.95]);C=['q05','q10','q50','q90','q95'];MIN=[52,39,20,39,52]

def maturity(frame,cutoff):
    cut=pd.Timestamp(cutoff)
    if cut.tzinfo is None:raise ValueError('Aware cutoff required')
    for k in ['decision_time','label_end','label_available_at']:
        d=pd.to_datetime(frame[k],utc=True)
        if (d>=pd.Timestamp('2024-01-01T00:00Z')).any():raise PermissionError('Reserved cohort')
    d=pd.to_datetime(frame.decision_time,utc=True);end=pd.to_datetime(frame.label_end,utc=True);at=pd.to_datetime(frame.label_available_at,utc=True)
    if (d>=cut).any() or (end>cut).any() or (at>cut).any() or (end<=d).any() or (at<end).any():raise PermissionError('Future/immature training')
    if frame.get('role',pd.Series(['OOF']*len(frame))).ne('OOF').any():raise PermissionError('Only genuine OOF labels fit calibration')
    return int(d.nunique())

def wquantile(x,p,w):
    x=np.asarray(x);w=np.asarray(w);order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    if not len(x) or not np.isfinite(x).all() or (w<0).any() or w.sum()<=0:raise ValueError('Quantile sample')
    return float(x[min(np.searchsorted(np.cumsum(w)/w.sum(),p,side='left'),len(x)-1)])

def fit(frame,kind,cutoff):
    if kind not in ['identity','intercept','cqr']:raise ValueError('Fixed calibration candidates only')
    dates=maturity(frame,cutoff);q=frame[C].to_numpy();y=frame.target.to_numpy();scale=frame.scale.to_numpy()
    if not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any() or not np.isfinite(y).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Calibration sample')
    counts=frame.groupby('decision_time').decision_time.transform('count');w=1/counts.to_numpy();support=[dates>=m and min(t,1-t)*dates>=20 for t,m in zip(Q,MIN)];offset=np.zeros(5)
    if kind=='intercept':
        for j,t in enumerate(Q):
            if support[j]:offset[j]=wquantile((y-q[:,j])/scale,t,w)
    if kind=='cqr':
        if support[2]:offset[2]=wquantile((y-q[:,2])/scale,.5,w)
        for lo,hi,coverage in [(0,4,.9),(1,3,.8)]:
            if support[lo] and support[hi]:
                score=np.maximum.reduce([(q[:,lo]-y)/scale,(y-q[:,hi])/scale,np.zeros(len(y))]);value=wquantile(score,coverage,w);offset[lo]=-value;offset[hi]=value
    if kind=='identity':support=[False]*5
    return {'kind':kind,'offset_normalized':offset.tolist(),'supported':support,'distinct_mature_dates':dates,'minimum_dates':MIN,'expected_smaller_tail_date_floor':20,'cutoff':pd.Timestamp(cutoff).isoformat(),'unsupported_reason':{str(Q[j]):'IDENTITY_INSUFFICIENT_DISTINCT_OR_EXPECTED_TAIL_DATES' for j in range(5) if not support[j]},'guaranteed_coverage':False,'outer_labels_used':False,'per_asset_calibrators':False}

def apply(q,scale,p):
    q=np.asarray(q,dtype=float);scale=np.asarray(scale,dtype=float)
    if q.shape!=(len(scale),5) or not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Inference schema')
    z=q+scale[:,None]*np.array(p['offset_normalized']);supported=np.asarray(p['supported'],bool);z[:,~supported]=q[:,~supported]
    # Project each supported contiguous segment inside fixed unsupported anchors.
    j=0
    while j<5:
        if not supported[j]:j+=1;continue
        start=j
        while j<5 and supported[j]:j+=1
        lo=q[:,start-1] if start else np.full(len(q),-np.inf);hi=q[:,j] if j<5 else np.full(len(q),np.inf)
        z[:,start:j]=np.clip(np.maximum.accumulate(z[:,start:j],axis=1),lo[:,None],hi[:,None])
    if (np.diff(z,axis=1)<0).any() or not np.isfinite(z).all():raise ValueError('Invalid calibrated forecast')
    return z

def loss(f,q):
    e=f.target.to_numpy()[:,None]-q;v=np.maximum(Q*e,(Q-1)*e).mean(1)/f.scale.to_numpy();g=pd.DataFrame({'date':f.decision_time,'v':v});return float(g.groupby('date').v.mean().mean())

def monitor(q,scale,thresholds,*,known_at,decision_time):
    if pd.Timestamp(known_at)>pd.Timestamp(decision_time):raise PermissionError('Future runtime input')
    q=np.asarray(q)
    if not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():return {'state':'RED_QUARANTINE','reason':'NUMERICAL_GUARD','live_notification_sent':False}
    width=(q[:,4]-q[:,0])/np.asarray(scale);limit=thresholds.get('width90_p99');return {'state':'AMBER_REVIEW' if limit is not None and (width>limit).any() else 'OBSERVED_NO_NORMATIVE_BREACH','operationally_frozen':False,'live_notification_sent':False,'threshold_source':'mature SIA OOF only'}