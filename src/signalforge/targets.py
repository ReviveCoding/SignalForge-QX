"""Calendar and corporate-action target oracle, separate from P0 price benchmarks."""
import pandas as pd
import numpy as np
from .portfolio import Ledger


def next_regular_open(decision,calendar='XNYS'):
    import exchange_calendars as xc
    t=pd.Timestamp(decision)
    if t.tzinfo is None:
        raise ValueError('Decision must be timezone aware')
    cal=xc.get_calendar(calendar,start=str((t-pd.Timedelta(days=7)).date()),end=str((t+pd.Timedelta(days=30)).date()))
    opens=cal.schedule['open']
    valid=opens[opens>t.tz_convert('UTC')]
    if valid.empty:
        raise ValueError('No regular session in calendar range')
    return valid.iloc[0]


def p1_total_return(prices,actions,start,end,asset='x',price_mode='P1'):
    if price_mode!='P1':
        raise ValueError('Execution target requires P1 raw prices plus actions')
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or start>=end:
        raise ValueError('Invalid target interval')
    frame=prices.copy()
    if any(pd.Timestamp(t).tzinfo is None for t in frame.time):raise ValueError('Raw price times must be aware')
    frame['time']=pd.to_datetime(frame.time,utc=True)
    if frame.time.duplicated().any() or 'adjusted' not in frame or frame.adjusted.any():
        raise ValueError('Raw unique prices required; adjusted/action double count prohibited')
    def mark(t):
        row=frame[frame.time.eq(t)]
        if len(row)!=1 or not isinstance(row.iloc[0].fresh,(bool,np.bool_)) or not bool(row.iloc[0].fresh):
            raise ValueError('Missing session or stale price')
        value=float(row.iloc[0].open)
        if not np.isfinite(value) or value<=0:raise ValueError('Nonpositive/nonfinite raw price')
        return value
    initial=mark(start)
    book=Ledger(0,{asset:1})
    priority={'split':0,'ex_dividend':1,'pay_dividend':2}
    if any(a['kind'] not in priority for a in actions):raise ValueError('Unknown corporate action')
    actions=sorted(actions,key=lambda a:(pd.Timestamp(a['time']),priority[a['kind']],a['id']))
    for action in actions:
        time=pd.Timestamp(action['time'])
        if time.tzinfo is None:
            raise ValueError('Action time must be aware')
        # Start-open buyers have no ex-date entitlement; prior holders at end-open do.
        if start<time<=end:
            if action['kind']=='split':
                book.split(asset,float(action['ratio']),action['id'])
            elif action['kind']=='ex_dividend':
                book.ex_dividend(asset,float(action['amount']),action['id'])
            elif action['kind']=='pay_dividend' and action['id'] in book.receivables:
                book.pay_dividend(action['id'])
            else:
                if action['kind'] not in {'pay_dividend'}:
                    raise ValueError('Unknown corporate action')
    return book.nav({asset:mark(end)})/initial-1
