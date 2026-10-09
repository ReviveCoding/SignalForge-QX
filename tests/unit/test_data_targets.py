import pandas as pd
import pytest
from signalforge.data import valid_mapping,validate_events,auxiliary_rows
from signalforge.targets import next_regular_open,p1_total_return


def test_calendar_holiday_dst():
    assert str(next_regular_open('2022-07-01T18:00:00-04:00'))=='2022-07-05 13:30:00+00:00'
    assert str(next_regular_open('2022-03-11T18:00:00-05:00'))=='2022-03-14 13:30:00+00:00'


def test_mapping_valid_and_known():
    m=pd.DataFrame({'entity':['x'],'mapped_entity':['y'],'valid_from':['2020-01-01T00:00Z'],
                    'valid_to':['2021-01-01T00:00Z'],'known_at':['2020-02-01T00:00Z']})
    with pytest.raises(ValueError):valid_mapping(m,'x','2020-01-02T00:00Z','2020-01-03T00:00Z')
    assert valid_mapping(m,'x','2020-01-02T00:00Z','2020-02-03T00:00Z')=='y'


def test_tier_a_rejects_unproven_claim():
    e=pd.DataFrame({'entity':['x'],'source':['eia'],'field':['stock'],'reference_time':['2020-01-01T00:00Z'],
                    'available_at':['2020-01-02T00:00Z'],'value':[0.],'unit':['barrels'],'raw_hash':['x'],'pit_tier':['A']})
    with pytest.raises(ValueError):validate_events(e)


def test_tier_a_hash_labels_do_not_prove_publication():
    e=pd.DataFrame({'entity':['x'],'source':['eia'],'field':['stock'],'reference_time':['2020-01-01T00:00Z'],
                    'available_at':['2020-01-02T00:00Z'],'value':[0.],'unit':['barrels'],'raw_hash':['a'*64],'pit_tier':['A'],
                    'public_evidence_hash':['b'*64],'original_vintage_hash':['c'*64]})
    with pytest.raises(ValueError,match='registry'):validate_events(e)
    e['pit_tier']='B';assert validate_events(e)


def prices():
    return pd.DataFrame({'time':['2022-01-03T14:30Z','2022-01-10T14:30Z'],'open':[10.,5.],
                         'fresh':[True,True],'adjusted':[False,False]})


def test_p1_split_and_dividend_oracle():
    actions=[{'time':'2022-01-05T14:30Z','kind':'split','ratio':2,'id':'s'},
             {'time':'2022-01-06T14:30Z','kind':'ex_dividend','amount':1,'id':'d'}]
    assert p1_total_return(prices(),actions,'2022-01-03T14:30Z','2022-01-10T14:30Z')==pytest.approx(.2)


def test_p1_missing_stale_adjusted_reject():
    p=prices();p.loc[1,'fresh']=False
    with pytest.raises(ValueError):p1_total_return(p,[],'2022-01-03T14:30Z','2022-01-10T14:30Z')
    p=prices();p['adjusted']=True
    with pytest.raises(ValueError):p1_total_return(p,[],'2022-01-03T14:30Z','2022-01-10T14:30Z')
    with pytest.raises(ValueError):p1_total_return(prices(),[],'2022-01-03T14:30Z','2022-01-10T14:30Z',price_mode='P0')


def test_ex_date_start_purchase_no_dividend():
    p=prices();p.loc[1,'open']=10
    actions=[{'time':'2022-01-03T14:30Z','kind':'ex_dividend','amount':1,'id':'d'}]
    assert p1_total_return(p,actions,'2022-01-03T14:30Z','2022-01-10T14:30Z')==0


def test_p1_invalid_initial_price_and_fresh_flag_rejected():
    for value in [0.,-1.,float('nan'),float('inf')]:
        p=prices();p.loc[0,'open']=value
        with pytest.raises(ValueError,match='raw price'):p1_total_return(p,[],'2022-01-03T14:30Z','2022-01-10T14:30Z')
    p=prices();p['fresh']='False'
    with pytest.raises(ValueError,match='stale'):p1_total_return(p,[],'2022-01-03T14:30Z','2022-01-10T14:30Z')


def test_same_session_split_dividend_action_order_invariant():
    actions=[{'time':'2022-01-05T14:30Z','kind':'ex_dividend','amount':1,'id':'d'}, {'time':'2022-01-05T14:30Z','kind':'split','ratio':2,'id':'s'}]
    one=p1_total_return(prices(),actions,'2022-01-03T14:30Z','2022-01-10T14:30Z')
    two=p1_total_return(prices(),actions[::-1],'2022-01-03T14:30Z','2022-01-10T14:30Z')
    assert one==two==pytest.approx(.2)
