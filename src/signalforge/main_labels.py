"""P1 Main/Nested fixed-grid labels with actual-session and publication lineage."""
import datetime,re
import numpy as np
import pandas as pd
from .targets import next_regular_open,p1_total_return
from .panels import matched_grid
from .runtime import digest


def build_main_labels(grid,prices,actions,horizon_rebalances=1,price_mode='P1',action_coverage=None):
    if price_mode!='P1':raise PermissionError('Main target requires P1; P0 needs a separate registered study')
    if horizon_rebalances not in {1,4}:raise ValueError('Registered one/four rebalance horizons required')
    # Deny reserved development origins before numerical prices/outcomes are read.
    decisions=pd.to_datetime(grid.decision_time,utc=True,format='mixed')
    if decisions.ge('2024-01-01T00:00Z').any():raise PermissionError('Development labels cannot read reserved targets')
    matched_grid(grid)
    for value in grid.decision_time:
        t=pd.Timestamp(value)
        if t.tzinfo is None:raise ValueError('Aware decision required')
        local=t.tz_convert('America/New_York')
        if local.weekday()!=4 or (local.hour,local.minute,local.second)!=(18,0,0):raise ValueError('Registered Friday 18:00 New York origins required')
    required={'asset','time','open','fresh','adjusted','available_at','raw_hash'}
    if not required<=set(prices):raise ValueError('P1 raw marks, publication clocks and hashes required')
    marks=prices.copy()
    for column in ['time','available_at']:
        if any(pd.isna(t) or pd.Timestamp(t).tzinfo is None for t in marks[column]):raise ValueError('Aware price clocks required')
        marks[column]=pd.to_datetime(marks[column],utc=True,format='mixed')
    if (marks.available_at<marks.time).any():raise ValueError('Raw open publication precedes actual session')
    if marks.time.ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved price rows cannot enter development labels')
    if marks.duplicated(['asset','time']).any():raise ValueError('Duplicate raw marks')
    if any(not isinstance(v,(bool,np.bool_)) for v in marks.adjusted) or marks.adjusted.any():raise ValueError('Typed unadjusted raw P1 marks required')
    if not marks.raw_hash.map(lambda h:isinstance(h,str) and bool(re.fullmatch('[a-f0-9]{64}',h))).all():raise ValueError('Price raw lineage required')
    for action in actions:
        if not {'asset','kind','time','id','available_at','raw_hash'}<=set(action):raise ValueError('Corporate-action source/publication lineage required')
        if pd.Timestamp(action['time']).tzinfo is None or pd.Timestamp(action['available_at']).tzinfo is None:raise ValueError('Aware corporate-action clocks required')
        if not re.fullmatch('[a-f0-9]{64}',action['raw_hash']):raise ValueError('Action raw lineage required')
    rows=[]
    for origin in grid.to_dict('records'):
        decision=pd.Timestamp(origin['decision_time']);local=decision.tz_convert('America/New_York')
        end_date=local.date()+datetime.timedelta(weeks=horizon_rebalances)
        next_decision=pd.Timestamp(end_date).tz_localize('America/New_York')+pd.Timedelta(hours=18)
        start=next_regular_open(decision);end=next_regular_open(next_decision)
        row={**origin,'price_mode':'P1','horizon_rebalances':horizon_rebalances,'label_start':start.isoformat(),'label_end':end.isoformat(),
             'y':np.nan,'label_available_at':None,'target_status':'MISSING_PRICE_OR_ACTION_EVIDENCE','economic_qualified':False,'target_raw_hashes':[]}
        # A 2023 origin requiring 2024 prices remains explicitly unlabelled.
        if end>=pd.Timestamp('2024-01-01T00:00Z'):
            row['target_status']='OUTSIDE_DEVELOPMENT_LABEL_BOUNDARY';rows.append(row);continue
        coverage=[c for c in (action_coverage or []) if c['asset']==origin['asset'] and c.get('complete') is True
                  and pd.Timestamp(c['start']).tzinfo is not None and pd.Timestamp(c['end']).tzinfo is not None
                  and pd.Timestamp(c['start'])<=start and pd.Timestamp(c['end'])>=end
                  and isinstance(c.get('raw_hash'),str) and re.fullmatch('[a-f0-9]{64}',c['raw_hash'])]
        if not coverage:
            row['target_status']='BLOCKED_ACTION_LEDGER_COVERAGE';rows.append(row);continue
        selected=marks[marks.asset.eq(origin['asset'])&marks.time.isin([start,end])]
        selected_actions=[a for a in actions if a['asset']==origin['asset'] and start<pd.Timestamp(a['time'])<=end]
        try:
            value=p1_total_return(selected,selected_actions,start,end,origin['asset'],'P1')
            publication=max([end,*selected.available_at.tolist(),*[pd.Timestamp(a['available_at']) for a in selected_actions]])
            if publication>=pd.Timestamp('2024-01-01T00:00Z'):
                row['target_status']='OUTSIDE_DEVELOPMENT_PUBLICATION_BOUNDARY'
            else:
                row.update(y=value,label_available_at=publication.isoformat(),target_status='ELIGIBLE_P1_TARGET',
                    target_raw_hashes=sorted(set(selected.raw_hash.tolist()+[a['raw_hash'] for a in selected_actions]+[c['raw_hash'] for c in coverage])))
        except ValueError as error:row['target_failure']=str(error)
        rows.append(row)
    out=pd.DataFrame(rows);out.attrs.update(grid.attrs)
    out.attrs['target_contract_id']=digest({'price_mode':'P1','horizon_rebalances':horizon_rebalances,'dividend_convention':'held_cash_and_receivables',
        'decision_clock':'Friday18AmericaNewYork','execution':'next_regular_XNYS_open','boundary':'development_before_2024'})
    out.attrs['economic_qualified']=False
    return out
