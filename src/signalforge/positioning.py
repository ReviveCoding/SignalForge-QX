"""Reported positioning in contracts; original publication clocks supplied explicitly."""
import numpy as np
import pandas as pd
from .pit import _aware_utc

FIELDS={'tff':['Dealer','Asset_Mgr','Lev_Money','Other_Rept'],
        'disaggregated':['Prod_Merc','Swap','M_Money','Other_Rept']}


def positioning_features(frame,family):
    if family not in FIELDS:raise ValueError('Unknown CFTC family')
    out=frame[['CFTC_Contract_Market_Code','report_date']].copy()
    oi=pd.to_numeric(frame.Open_Interest_All,errors='raise')
    if np.isinf(oi).any() or (oi<0).any():raise ValueError('Invalid open interest')
    out['open_interest']=oi;out['zero_or_missing_open_interest']=oi.isna()|oi.eq(0)
    denominator=oi.where(oi>0)
    for category in FIELDS[family]:
        # Official disaggregated schema uses a doubled underscore for swap shorts.
        long=category+'_Positions_Long_All';short=('Swap__' if category=='Swap' else category+'_')+'Positions_Short_All'
        if not {long,short}<=set(frame):raise ValueError('CFTC category schema missing: '+category)
        a,b=pd.to_numeric(frame[long],errors='raise'),pd.to_numeric(frame[short],errors='raise')
        if np.isinf(a).any() or np.isinf(b).any() or (a<0).any() or (b<0).any():raise ValueError('Invalid gross positions')
        out[category+'_net_contracts']=a-b
        out[category+'_net_fraction_open_interest']=(a-b)/denominator
    return out


def positioning_events(features,raw_hash,release_evidence):
    """Never replace actual release exceptions with nominal Friday timestamps."""
    rows=[];blocked=[]
    for record in features.to_dict('records'):
        date=pd.Timestamp(record['report_date']).date().isoformat()
        evidence=release_evidence.get(date)
        if not evidence or not evidence.get('clock_evidence_hash'):
            blocked.append({'report_date':date,'contract':record['CFTC_Contract_Market_Code'],'reason':'ACTUAL_RELEASE_CLOCK_ABSENT'});continue
        known=_aware_utc([evidence['available_at']],'CFTC actual public release').iloc[0]
        reference=pd.Timestamp(date,tz='UTC')
        if known<reference:raise ValueError('Release precedes reference')
        for field,value in record.items():
            if field in {'report_date','CFTC_Contract_Market_Code','zero_or_missing_open_interest'}:continue
            rows.append({'entity':record['CFTC_Contract_Market_Code'],'source':'cftc','field':field,'value':value,
                         'reference_time':reference.isoformat(),'available_at':known.isoformat(),'raw_hash':raw_hash,
                         'pit_tier':'B','clock_evidence_hash':evidence['clock_evidence_hash'],
                         'unit':'fraction_open_interest' if 'fraction' in field else 'contracts','positioning_is_cash_flow':False,
                         'original_vintage_qualified':False})
    return pd.DataFrame(rows),blocked
