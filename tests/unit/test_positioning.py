import numpy as np
import pandas as pd
import pytest
from signalforge.positioning import positioning_features,positioning_events,FIELDS


def test_positions_missing_and_zero_oi_never_cashflow():
    frame=pd.DataFrame({'CFTC_Contract_Market_Code':['x','y'],'report_date':['2020-01-07','2020-01-07'],'Open_Interest_All':[100,0]})
    for category in FIELDS['tff']:
        frame[category+'_Positions_Long_All']=[30,0];frame[category+'_Positions_Short_All']=[10,0]
    features=positioning_features(frame,'tff')
    assert features.iloc[0].Dealer_net_fraction_open_interest==.2
    assert np.isnan(features.iloc[1].Dealer_net_fraction_open_interest)
    rows,blocked=positioning_events(features,'a'*64,{})
    assert rows.empty and len(blocked)==2
    rows,blocked=positioning_events(features,'a'*64,{'2020-01-07':{'available_at':'2020-01-10T20:30Z','clock_evidence_hash':'b'*64}})
    assert not blocked and rows.pit_tier.eq('B').all() and not rows.positioning_is_cash_flow.any()
