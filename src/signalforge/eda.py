"""Outcome-blind canonical-event QA, bounded strictly before reserved dates."""
import numpy as np
import pandas as pd
from .data import validate_events
from .runtime import digest


def event_audit(events,reserved_start='2024-01-01T00:00Z'):
    frame=events.copy();cut=pd.Timestamp(reserved_start)
    for name in ['reference_time','available_at']:
        frame[name]=pd.to_datetime(frame[name],utc=True)
        if (frame[name]>=cut).any():raise PermissionError('Raw EDA cannot expose reserved observations')
    validate_events(events)
    order=['entity','source','field','reference_time','available_at','raw_hash']
    canonical=frame.sort_values(order).reset_index(drop=True)
    columns=sorted(canonical.columns)
    content=canonical[columns].replace({np.nan:None}).astype(object)
    for name in ['reference_time','available_at']:content[name]=canonical[name].astype(str)
    series=[]
    for keys,group in canonical.groupby(['entity','source','field','unit'],dropna=False,sort=True):
        reference_public=(group.available_at-group.reference_time).dt.total_seconds()/86400
        versions=group.sort_values(['reference_time','available_at']).copy()
        first=versions.groupby('reference_time').available_at.transform('min')
        revision_lag=(versions.available_at-first).dt.total_seconds()/86400
        value=pd.to_numeric(group.value)
        series.append(dict(zip(['entity','source','field','unit'],keys))|{
            'rows':len(group),'missing':int(value.isna().sum()),'true_zero':int(value.eq(0).sum()),
            'distinct_reference_dates':int(group.reference_time.nunique()),
            'revisions':int(group.duplicated('reference_time').sum()),
            'reference_to_public_days':{'min':float(reference_public.min()),'median':float(reference_public.median()),'max':float(reference_public.max())},
            'revision_from_first_public_days':{'median':float(revision_lag.median()),'max':float(revision_lag.max())},
            'extreme_values_retained':True,'minimum':float(value.min()) if value.notna().any() else None,
            'maximum':float(value.max()) if value.notna().any() else None})
    return {'state':'SUCCEEDED_RECONSTRUCTED_RAW_QA','canonical_content_id':digest(content.to_dict('records')),
            'rows':len(frame),'series':series,'reserved_access':False,'tier_a_qualified':False,
            'clock_claim':'Observed reconstructed archive issue bounds; no original-vintage authentication',
            'release_lag_is_not_retrieval_lag':True,'outlier_deletions':0}
