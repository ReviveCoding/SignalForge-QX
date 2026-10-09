"""Shared calendar-block indices, segmented gaps and FP64 paired diagnostics."""
import numpy as np
import pandas as pd
from .runtime import digest


def calendar_indices(dates,block=8,draws=2000,seed=731):
    dates=pd.DatetimeIndex(pd.to_datetime(dates,utc=True));n=len(dates)
    if not n or dates.has_duplicates or not dates.is_monotonic_increasing or block<1 or draws<100:
        raise ValueError('Unique ordered calendar dates and registered draws required')
    gaps=np.flatnonzero(np.diff(dates.asi8)>10*86400*10**9)+1
    segments=np.split(np.arange(n),gaps);rng=np.random.default_rng(seed);output=[]
    for segment in segments:
        width=min(block,len(segment));count=int(np.ceil(len(segment)/width))
        starts=rng.integers(0,len(segment),size=(draws,count))
        local=((starts[:,:,None]+np.arange(width))%len(segment)).reshape(draws,-1)[:,:len(segment)]
        output.append(segment[local])
    indices=np.concatenate(output,axis=1)
    return indices,{'n_dates':n,'segment_lengths':[len(s) for s in segments],'block_weeks':block,'draws':draws,'seed':seed,
                    'index_id':digest({'dates':dates.astype(str).tolist(),'block':block,'draws':draws,'seed':seed}),
                    'underpowered':n<26 or max(map(len,segments))<2*block,
                    'assumptions':'Within-segment block stationarity approximation; calendar gaps never bridged'}


def paired_statistics(left,right,indices,metadata):
    left,right=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if left.shape!=right.shape or left.ndim!=1 or indices.shape[1]!=len(left) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Aligned finite date-level losses and shared indices required')
    delta=left-right;sampled=delta[indices].mean(1);effect=float(delta.mean())
    centered=(delta-effect)[indices].mean(1)
    p=(1+np.sum(np.abs(centered)>=abs(effect)))/(len(centered)+1)
    se=float(sampled.std(ddof=1));lower,upper=np.quantile(sampled,[.025,.975])
    return {**metadata,'mean_improvement':effect,'relative_gain':effect/float(left.mean()) if left.mean()!=0 else None,
            'lower_95':float(lower),'upper_95':float(upper),'two_sided_centered_block_p':float(p),
            'approximate_unadjusted_MDE_80':float((1.96+.8416)*se),'standard_error':se,
            'MDE_scope':'Development diagnostic only; thresholds never selected from final MDE',
            'equivalence_established':False,
            'nonsignificance_interpretation':'Absence of detected improvement does not establish equivalence; no equivalence margin/test was registered',
            'n_training_seeds_are_independent_market_samples':False}


def exploratory_fdr(pvalues):
    """Complete exploratory family; missing contrasts keep multiplicity size."""
    keys=list(pvalues);n=len(keys)
    if not n:raise ValueError('Full nonempty exploratory family required')
    values=np.array([1. if pvalues[k] is None else pvalues[k] for k in keys])
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Valid full-family p-values required')
    order=np.argsort(values,kind='stable');scaled=values[order]*n/np.arange(1,n+1)
    adjusted=np.minimum.accumulate(scaled[::-1])[::-1].clip(0,1)
    harmonic=np.sum(1/np.arange(1,n+1))
    bh={keys[i]:None if pvalues[keys[i]] is None else float(adjusted[j]) for j,i in enumerate(order)}
    by={keys[i]:None if pvalues[keys[i]] is None else float(min(1.,adjusted[j]*harmonic)) for j,i in enumerate(order)}
    return {'BH':bh,'BY_arbitrary_dependence':by,'family_size':n,'registered_family':keys,
        'scope':'exploratory; not confirmatory Holm','assumptions':'BH requires valid p-values and independence/PRDS; common-baseline temporal dependence need not satisfy PRDS. BY is a conservative dependence sensitivity, conditional on valid block p-values.'}
