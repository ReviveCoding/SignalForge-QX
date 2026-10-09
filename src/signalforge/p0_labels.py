"""Separate weekly P0 prediction protocol; never execution-grade economics."""
import datetime,re
import numpy as np
import pandas as pd
from .runtime import digest
from .panels import matched_grid


def last_regular_close(origin):
    import exchange_calendars as xc
    t=pd.Timestamp(origin)
    cal=xc.get_calendar('XNYS',start=str((t-pd.Timedelta(days=14)).date()),end=str((t+pd.Timedelta(days=7)).date()))
    closes=cal.schedule['close'];return closes[closes<=t].iloc[-1]


def build_p0_labels(grid,weekly_marks,basis,horizon_rebalances=1):
    if basis not in {'unadjusted_close_price_return','vendor_adjusted_close_benchmark'}:raise ValueError('Explicit P0 close basis required')
    if horizon_rebalances not in {1,4}:raise ValueError('Registered P0 horizon required')
    matched_grid(grid)
    origins=pd.to_datetime(grid.decision_time,utc=True,format='mixed')
    if origins.ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved P0 origins forbidden')
    marks=weekly_marks.copy()
    if not {'asset','decision_time','time','available_at','close','fresh','adjusted','raw_hash'}<=set(marks):raise ValueError('Full P0 publication/mark lineage required')
    for column in ['decision_time','time','available_at']:
        if any(pd.isna(t) or pd.Timestamp(t).tzinfo is None for t in marks[column]):raise ValueError('Aware P0 mark clocks required')
        marks[column]=pd.to_datetime(marks[column],utc=True,format='mixed')
        if marks[column].ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved P0 marks forbidden')
    if marks.duplicated(['asset','decision_time']).any():raise ValueError('Unique P0 marks required')
    expected=basis=='vendor_adjusted_close_benchmark'
    if any(not isinstance(v,(bool,np.bool_)) for v in marks.adjusted) or not marks.adjusted.eq(expected).all():raise ValueError('No silent P0 adjustment-basis switch')
    if not marks.raw_hash.map(lambda h:isinstance(h,str) and bool(re.fullmatch('[a-f0-9]{64}',h))).all():raise ValueError('P0 raw lineage required')
    output=[]
    for row in grid.to_dict('records'):
        origin=pd.Timestamp(row['decision_time']);local=origin.tz_convert('America/New_York')
        if local.weekday()!=4 or (local.hour,local.minute,local.second)!=(18,0,0):raise ValueError('Friday18 New York P0 origin required')
        future=pd.Timestamp(local.date()+datetime.timedelta(weeks=horizon_rebalances)).tz_localize('America/New_York')+pd.Timedelta(hours=18)
        result={**row,'price_mode':'P0','p0_basis':basis,'horizon_rebalances':horizon_rebalances,'label_start':None,
            'label_end':None,'label_available_at':None,'y':np.nan,'economic_qualified':False,
            'target_status':'MISSING_P0_MARK','target_raw_hashes':[],'p0_origin_mark_available_at':None}
        if future>=pd.Timestamp('2024-01-01T00:00Z'):result['target_status']='OUTSIDE_DEVELOPMENT_LABEL_BOUNDARY';output.append(result);continue
        selected=[]
        for decision in [origin,future]:
            match=marks[marks.asset.eq(row['asset'])&marks.decision_time.eq(decision)]
            if len(match)!=1:break
            mark=match.iloc[0]
            if (not isinstance(mark.fresh,(bool,np.bool_)) or not mark.fresh or
                not mark.time<=mark.available_at<=decision or not np.isfinite(float(mark.close)) or float(mark.close)<=0):break
            selected.append(mark)
        if len(selected)==2:
            left,right=selected
            if right.time<=left.time:raise ValueError('P0 target mark must advance after origin mark')
            result.update(label_start=left.time.isoformat(),label_end=right.time.isoformat(),
                y=float(right.close/left.close-1),label_available_at=right.available_at.isoformat(),
                p0_origin_mark_available_at=left.available_at.isoformat(),target_status='ELIGIBLE_P0_FORECAST_TARGET',
                target_raw_hashes=sorted({left.raw_hash,right.raw_hash}))
        output.append(result)
    panel=pd.DataFrame(output);panel.attrs.update(grid.attrs)
    panel.attrs['target_contract_id']=digest({'price_mode':'P0','basis':basis,'horizon_rebalances':horizon_rebalances,'economics':False})
    return panel
