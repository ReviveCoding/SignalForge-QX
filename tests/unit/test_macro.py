import json
import pandas as pd
import pytest
from signalforge.macro import request_plan,acquire_series
from signalforge.sources import parse_fred


def test_requests_never_default_to_latest_or_reserved():
    plan=request_plan('CPIAUCSL')
    assert [p['params']['output_type'] for p in plan]==[1,3]
    assert all(p['params']['realtime_end']<='2023-12-31' for p in plan)
    with pytest.raises(PermissionError):request_plan('CPIAUCSL',end='2024-01-01')


def test_month_reference_and_conservative_vintage_clock(tmp_path):
    path=tmp_path/'macro.json';path.write_text(json.dumps({'observations':[{'date':'2020-01-01','realtime_start':'2020-02-05','realtime_end':'2020-03-03','value':'1.2'}]}))
    event=parse_fred(path,'M')[0]
    assert event['reference_time']=='2020-01-31T00:00:00+00:00'
    assert event['available_at']=='2020-02-06T12:00:00+00:00' and event['pit_tier']=='B'


def test_missing_credential_no_request(monkeypatch):
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    with pytest.raises(PermissionError,match='BLOCKED_AUTH'):acquire_series(None,'DGS2')


def test_macro_units_bound_to_metadata_vintage_and_hash(tmp_path):
    from signalforge.macro import canonical_macro_events,metadata_request
    from signalforge.runtime import file_hash
    raw={'reference_time':'2020-01-31T00:00Z','available_at':'2020-02-06T12:00Z','realtime_start':'2020-02-05','value':1.,'raw_hash':'a'*64,'pit_tier':'B'}
    canonical,blocked=canonical_macro_events('CPIAUCSL',[raw],{})
    assert not canonical and blocked[0]['state']=='BLOCKED_UNIT_VINTAGE'
    path=tmp_path/'metadata.json';path.write_text(json.dumps({'seriess':[{'id':'CPIAUCSL','frequency_short':'M','units':'Index 1982-1984=100','realtime_start':'2020-02-05','realtime_end':'2020-02-05'}]}))
    evidence={'2020-02-05':{'path':str(path),'sha256':file_hash(path),'source_url':metadata_request('CPIAUCSL','2020-02-05')['url'],'public_available_at':'2020-02-05T12:00Z'}}
    canonical,blocked=canonical_macro_events('CPIAUCSL',[raw],evidence)
    assert not blocked and canonical[0]['source']=='macro' and canonical[0]['unit']=='Index 1982-1984=100'
    evidence['2020-02-05']['public_available_at']='2020-02-07T00:00Z'
    with pytest.raises(ValueError,match='not known'):canonical_macro_events('CPIAUCSL',[raw],evidence)
    with pytest.raises(PermissionError):metadata_request('CPIAUCSL','2024-01-01')
