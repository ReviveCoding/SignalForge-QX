"""Additional local lineage/session tests; public evidence is never generated here."""
import json
from pathlib import Path
import pandas as pd
import pytest
from signalforge.tiingo import parse_tiingo,canonical_inputs,SYMBOLS,reconstructed_eod_available_at,request,new_plan
from signalforge.runtime import file_hash
from signalforge.sources import Acquisition

REPO=Path(__file__).parents[2]


def test_local_fixture_raw_prices_and_action_claim_boundary():
    fixture=REPO/'tests/fixtures/synthetic/tiingo_eod.json'
    records=[r for asset in SYMBOLS for r in parse_tiingo(fixture,asset)]
    origins=['2022-04-15T18:00:00-04:00','2022-04-22T18:00:00-04:00']
    marks,events,p1=canonical_inputs(records,origins)
    assert len(marks)==8 and marks[0]['close']==102 and marks[0]['adjusted'] is False
    assert all(m['time']=='2022-04-14T20:00:00+00:00' for m in marks)
    assert all(m['available_at']=='2022-04-15T00:30:00+00:00' for m in marks)
    assert all(e['pit_tier']=='B' and not e['original_publication_qualified'] for e in events)
    assert not p1['economic_qualified'] and not p1['qualified_for_final']
    assert p1['actions'][0]['divCash']==.5 and p1['actions'][0]['pay_date'] is None


def test_twenty_session_volatility_and_four_feature_units(tmp_path):
    import exchange_calendars as xc
    days=xc.get_calendar('XNYS',start='2022-02-01',end='2022-04-30').schedule.index
    rows=[]
    for i,day in enumerate(days):
        value=100+i
        row={k:float(value) for k in ['open','high','low','close','adjOpen','adjHigh','adjLow','adjClose']}
        row.update(date=day.date().isoformat()+'T00:00:00.000Z',volume=1000,adjVolume=1000,divCash=0,splitFactor=1)
        rows.append(row)
    path=tmp_path/'daily.json';path.write_text(json.dumps(rows))
    records=[r for asset in SYMBOLS for r in parse_tiingo(path,asset)]
    origins=[(d.tz_localize('America/New_York')+pd.Timedelta(hours=18)).isoformat() for d in pd.date_range('2022-02-04','2022-04-29',freq='W-FRI')]
    _,events,_=canonical_inputs(records,origins)
    assert {(e['field'],e['unit']) for e in events}=={('return_1w','fraction'),('return_4w','fraction'),
        ('realized_vol_20','daily_fraction'),('log_volume','log1p_shares')}
    assert all(e['raw_hash']==file_hash(path) and pd.Timestamp(e['available_at'])==reconstructed_eod_available_at(pd.Timestamp(e['reference_time'])) for e in events)
    records=[r for r in records if r['session_date']!='2022-04-13']
    _,events,_=canonical_inputs(records,origins)
    affected=[e for e in events if e['field']=='realized_vol_20' and e['reference_time']>='2022-04-13']
    assert affected and all(e['value'] is None for e in affected)


def test_calendar_grid_rejects_favorable_intersection_and_dst_wrong_origin():
    with pytest.raises(ValueError,match='Contiguous'):
        canonical_inputs([],['2022-04-08T18:00:00-04:00','2022-04-22T18:00:00-04:00'])
    with pytest.raises(ValueError,match='Friday'):
        canonical_inputs([],['2022-03-18T18:00:00-05:00'])


def test_missing_origin_explicit_events_prevent_stale_feature_reuse(tmp_path):
    from signalforge.panels import fixed_grid,build_features,FeatureSpec
    fixture=REPO/'tests/fixtures/synthetic/tiingo_eod.json'
    records=parse_tiingo(fixture,'SPY')
    origins=['2022-04-15T18:00:00-04:00','2022-04-22T18:00:00-04:00']
    _,events,_=canonical_inputs(records,origins)
    missing=[e for e in events if e['reference_time']=='2022-04-21T20:00:00+00:00']
    assert len(missing)==4 and all(e['value'] is None for e in missing)
    grid=fixed_grid(origins,{'assets':['SPY'],'outcome_blind':True,'frozen_before_outcomes':'synthetic only'})
    panel=build_features(grid,pd.DataFrame(events),[FeatureSpec('volume','market','log_volume','ASSET','log1p_shares')],['market'])
    assert panel.x.iloc[-1][3:5]==[0.,0.]


def test_header_token_and_redirect_do_not_expose_or_forward_secret(tmp_path,monkeypatch):
    monkeypatch.setenv('TIINGO_API_TOKEN','synthetic-secret');calls=[]
    c=Acquisition(tmp_path)
    class Response:
        status_code=302;headers={'Location':'https://api.tiingo.com/tiingo/daily/SPY'}
        def close(self):pass
    def get(url,**kwargs):calls.append((url,kwargs));return Response()
    monkeypatch.setattr(c.session,'get',get)
    url,params=request('SPY',new_plan())
    with pytest.raises(ValueError,match='redirect'):c.get(url,params)
    assert len(calls)==1 and 'synthetic-secret' not in calls[0][0]
    assert calls[0][1]['headers']['Authorization']=='Token synthetic-secret'


def test_friday_18_origin_excludes_same_day_final_eod_close(tmp_path):
    rows=[]
    for day,close in [('2022-04-21',100.),('2022-04-22',110.)]:
        rows.append({'date':day+'T00:00:00.000Z','open':close,'high':close,'low':close,'close':close,
            'volume':1000,'adjOpen':close,'adjHigh':close,'adjLow':close,'adjClose':close,'adjVolume':1000,
            'divCash':0,'splitFactor':1})
    path=tmp_path/'two_days.json';path.write_text(json.dumps(rows))
    records=[r for asset in SYMBOLS for r in parse_tiingo(path,asset)]
    marks,events,_=canonical_inputs(records,['2022-04-22T18:00:00-04:00'])
    assert len(marks)==len(SYMBOLS)
    assert all(m['time']=='2022-04-21T20:00:00+00:00' for m in marks)
    assert all(m['available_at']=='2022-04-22T00:30:00+00:00' for m in marks)
    assert all(pd.Timestamp(e['available_at'])<=pd.Timestamp('2022-04-22T18:00:00-04:00') for e in events)
