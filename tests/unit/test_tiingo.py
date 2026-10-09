import copy, json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest

from signalforge.runtime import file_hash
from signalforge.sources import request_identity, redact_url
from signalforge.tiingo import (
    SYMBOLS, START, END, new_plan, validate_plan, validate_tiingo_url,
    request, parse_tiingo, canonical_inputs, build_inputs, reconstructed_eod_available_at,
)


def row(day='2023-12-29', close=100.0):
    return {
        'date': day+'T00:00:00.000Z',
        'open': close-1, 'high': close+1, 'low': close-2, 'close': close,
        'volume': 1000000,
        'adjOpen': close-1, 'adjHigh': close+1, 'adjLow': close-2, 'adjClose': close,
        'adjVolume': 1000000,
        'divCash': 0.0, 'splitFactor': 1.0,
    }


def write_rows(tmp_path, rows):
    p=tmp_path/'rows.json'
    p.write_text(json.dumps(rows))
    return p


def test_plan_and_url_are_exact_bounded_and_secret_free():
    p=new_plan(); validate_plan(p)
    assert p['symbols']==SYMBOLS and p['startDate']==START and p['endDate']==END
    assert not any(k.lower() in {'token','key','api_key','apikey'} for k in p)
    url,params=request('SPY',p)
    assert validate_tiingo_url(url+'?'+urlencode(params))=='SPY'
    for bad in [
        url.replace('api.tiingo.com','example.com')+'?'+urlencode(params),
        'https://api.tiingo.com/tiingo/daily/SPY?'+urlencode(params),
        url+'?'+urlencode({**params,'endDate':'2024-01-02'}),
        'https://api.tiingo.com/tiingo/daily/NOTREAL/prices?'+urlencode(params),
    ]:
        with pytest.raises(PermissionError): validate_tiingo_url(bad)


def test_request_identity_never_depends_on_secret():
    p=new_plan(); url,params=request('SPY',p)
    one=request_identity(url,{**params,'token':'AAA','api_key':'BBB'})
    two=request_identity(url,{**params,'token':'CCC','api_key':'DDD'})
    assert one==two
    rendered=redact_url(url+'?'+urlencode({**params,'token':'SECRET'}))
    assert 'SECRET' not in rendered and 'REDACTED' in rendered


def test_parser_rejects_reserved_duplicate_unordered_negative_and_metadata(tmp_path):
    p=new_plan()
    good=write_rows(tmp_path,[row('2023-12-28',99),row('2023-12-29',100)])
    parsed=parse_tiingo(good,'SPY',p,file_hash(good))
    assert len(parsed)==2 and all(x['asset']=='SPY' for x in parsed)
    assert all(x['raw_price_basis']=='unadjusted' for x in parsed)

    cases=[
        [row('2024-01-02')],
        [row('2023-12-29'),row('2023-12-29')],
        [row('2023-12-29'),row('2023-12-28')],
        [{**row('2023-12-29'),'close':-1}],
        [{**row('2023-12-29'),'regularMarketPrice':123}],
    ]
    for i,rows in enumerate(cases):
        q=tmp_path/f'bad{i}.json'; q.write_text(json.dumps(rows))
        with pytest.raises((PermissionError,ValueError)): parse_tiingo(q,'SPY',p)


def test_canonical_p0_keeps_weekly_grid_and_reconstructed_clock(tmp_path):
    import exchange_calendars as xc
    origins=(pd.date_range('2023-12-15','2023-12-29',freq='W-FRI',tz='America/New_York')+
             pd.Timedelta(hours=18))
    cal=xc.get_calendar('XNYS',start='2023-12-01',end='2023-12-31')
    records=[]
    for origin,close_value in zip(origins,[100.,101.,102.]):
        eligible=[close for close in cal.schedule['close'] if reconstructed_eod_available_at(close)<=origin.tz_convert('UTC')]
        close=eligible[-1]
        r=row(str(close.date()),close_value)
        r.update(session_date=str(close.date()),asset='SPY',raw_hash='a'*64)
        records.append(r)
    marks,events,p1=canonical_inputs(records,[x.isoformat() for x in origins])
    assert len(marks)==3
    assert {m['decision_time'] for m in marks}=={x.isoformat() for x in origins}
    assert all(m['pit_tier']=='B' and not m['original_publication_qualified'] for m in marks)
    assert all(pd.Timestamp(m['available_at'])>pd.Timestamp(m['time']) for m in marks)
    assert any(e['field']=='return_1w' for e in events)
    assert p1['economic_qualified'] is False and p1['state']=='BLOCKED_ACTION_CLOCK_AUDIT'


def test_builder_requires_real_immutable_acquisition(tmp_path):
    repo=tmp_path/'repo'; runtime=tmp_path/'runtime'
    (repo/'.local').mkdir(parents=True); runtime.mkdir()
    with pytest.raises(RuntimeError,match='real bounded Tiingo receipt absent'):
        build_inputs(repo,runtime)


def test_credential_helper_is_process_only_and_never_persists_values():
    root=Path(__file__).parents[2]
    text=(root/'scripts/Set-SignalForgeCredentials.ps1').read_text()
    assert "SetEnvironmentVariable($credentialName,$credentialText,'Process')" in text
    forbidden=['Set-ItemProperty','New-ItemProperty','setx ','Registry::','Machine\')']
    assert not any(token in text for token in forbidden)
    assert "Write-Host ('{0}: {1}'" in text
    assert 'Write-Host $credentialText' not in text
