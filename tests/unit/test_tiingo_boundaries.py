"""Additional synthetic boundary fixtures; public research evidence remains separate."""
import json
from pathlib import Path
from urllib.parse import urlencode
import pandas as pd
import pytest
from signalforge.runtime import atomic_json,commit_bundle,digest,file_hash
from signalforge.tiingo import (RAW_FIELDS,SYMBOLS,START,END,new_plan,validate_plan,
    register_plan,request,parse_tiingo,canonical_inputs,build_inputs,validate_tiingo_url)
from signalforge.sources import request_identity,validate_sec_user_agent,Acquisition

REPO=Path(__file__).parents[2]

def row(day='2022-04-14'):
    return {'date':day+'T00:00:00Z','open':100,'high':102,'low':99,'close':101,
            'volume':1000,'divCash':0,'splitFactor':1}

def test_raw_fields_required_adjusted_fields_optional(tmp_path):
    path=tmp_path/'raw.json';atomic_json(path,[row()])
    parsed=parse_tiingo(path,'SPY')
    assert parsed[0]['raw_price_basis']=='unadjusted' and parsed[0]['adjusted_fields_present']==[]
    atomic_json(path,[{**row(),'adjClose':51}])
    assert parse_tiingo(path,'SPY')[0]['adjClose']==51
    for field in RAW_FIELDS:
        bad=row();del bad[field];atomic_json(path,[bad])
        with pytest.raises(ValueError):parse_tiingo(path,'SPY')

@pytest.mark.parametrize('field',['open','high','low','close','volume','divCash','splitFactor','adjVolume'])
@pytest.mark.parametrize('value',[-1,float('nan'),float('inf'),True])
def test_invalid_numeric_values(tmp_path,field,value):
    path=tmp_path/'raw.json';path.write_text(json.dumps([{**row(),field:value}]))
    with pytest.raises(ValueError):parse_tiingo(path,'SPY')

def test_unordered_dates_and_unknown_symbol(tmp_path):
    path=tmp_path/'raw.json';atomic_json(path,[row('2022-04-18'),row()])
    with pytest.raises(ValueError,match='monotonic'):parse_tiingo(path,'SPY')
    with pytest.raises(ValueError,match='symbol'):parse_tiingo(path,'AAPL')

@pytest.mark.parametrize('suffix',[
    '/tiingo/daily/AAPL/prices','/tiingo/daily/spy/prices','/tiingo/daily/SPY/prices/',
    '/tiingo/daily/%53PY/prices','/tiingo/daily/SPY/../QQQ/prices'])
def test_exact_endpoint_paths(suffix):
    params={k:new_plan()[k] for k in ['startDate','endDate','resampleFreq','format']}
    with pytest.raises(PermissionError):validate_tiingo_url('https://api.tiingo.com'+suffix+'?'+urlencode(params))

def test_url_credentials_and_duplicates_rejected_identity_token_independent():
    url,params=request('SPY',new_plan());bounded=url+'?'+urlencode(params)
    assert validate_tiingo_url(bounded)=='SPY'
    for suffix in ['&token=fixture','&endDate=2023-12-31','#metadata']:
        with pytest.raises(PermissionError):validate_tiingo_url(bounded+suffix)
    assert request_identity(bounded+'&token=one')==request_identity(bounded+'&token=two')

def test_transport_header_not_receipt_or_identity(tmp_path,monkeypatch):
    monkeypatch.setenv('TIINGO_API_TOKEN','fixture-token');client=Acquisition(tmp_path)
    url,params=request('SPY',new_plan());captured=[]
    class Response:
        status_code=200
    def get(url,**kwargs):captured.append(kwargs);return Response()
    monkeypatch.setattr(client.session,'get',get)
    identity=request_identity(url,params)
    client.get(url,params)
    assert captured[0]['headers']['Authorization']=='Token fixture-token'
    monkeypatch.setenv('TIINGO_API_TOKEN','different-fixture')
    assert request_identity(url,params)==identity
    assert not list((tmp_path/'raw').rglob('*.receipt.json'))

def test_stream_error_never_exposes_key_or_commits_partial(tmp_path,monkeypatch):
    import requests
    client=Acquisition(tmp_path)
    class Response:
        headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,*args):raise requests.RequestException('fixture-secret-url');yield b''
    monkeypatch.setattr(client,'get',lambda *a,**k:Response())
    with pytest.raises(RuntimeError) as error:client.fetch('https://api.stlouisfed.org/fred/series/observations','json',{'api_key':'fixture-secret'})
    assert 'fixture-secret' not in str(error.value)
    assert not list((tmp_path/'raw').rglob('*.receipt.json')) and not list((tmp_path/'raw').rglob('*.partial'))

def test_grid_gaps_forbidden_and_clock_after_actual_close(tmp_path):
    path=tmp_path/'raw.json';atomic_json(path,[row()]);records=parse_tiingo(path,'SPY')
    origins=['2022-04-15T18:00:00-04:00','2022-04-22T18:00:00-04:00']
    marks,events,p1=canonical_inputs(records,origins)
    assert marks[0]['time']=='2022-04-14T20:00:00+00:00'
    assert pd.Timestamp(marks[0]['available_at'])>pd.Timestamp(marks[0]['time'])
    assert p1['state']=='BLOCKED_ACTION_CLOCK_AUDIT' and not p1['economic_qualified']
    with pytest.raises(ValueError,match='grid'):canonical_inputs(records,[origins[0],'2022-04-29T18:00:00-04:00'])

def test_forged_summary_without_immutable_bundle_cannot_create_manifests(tmp_path):
    plan=register_plan(tmp_path,tmp_path/'runtime')
    atomic_json(tmp_path/'reports/tiingo_development_acquisition.json',{
        'plan_id':validate_plan(plan),'state':'SUCCEEDED_RAW_TIINGO_DEVELOPMENT','artifact_id':'a'*64})
    with pytest.raises(FileNotFoundError):build_inputs(tmp_path,tmp_path/'runtime')
    assert not list((tmp_path/'.local').glob('inputs_*.json'))

def test_committed_fixture_result_without_live_receipts_refused(tmp_path):
    runtime=tmp_path/'runtime';plan=register_plan(tmp_path,runtime)
    result={'state':'SUCCEEDED_RAW_TIINGO_DEVELOPMENT','plan_id':validate_plan(plan),
        'evidence_kind':'synthetic_fixture','reserved_access':False,'raw_receipts':[],'records':[]}
    identity=digest(result)
    commit_bundle(runtime/'artifacts/tiingo_request_results'/identity,{'results.json':result},{})
    atomic_json(tmp_path/'reports/tiingo_development_acquisition.json',{
        'plan_id':validate_plan(plan),'state':result['state'],'artifact_id':identity})
    with pytest.raises(PermissionError):build_inputs(tmp_path,runtime)
    assert not list((tmp_path/'.local').glob('inputs_*.json'))

@pytest.mark.parametrize('value',['','SignalForge','SignalForge contact@example.com','Name x@invalid.test','Name\r\nx@organization.org'])
def test_sec_placeholder_or_invalid_contact_denied_before_network(tmp_path,monkeypatch,value):
    from signalforge.sec_pilot import acquire_nport_pilot
    monkeypatch.setenv('SEC_USER_AGENT',value)
    with pytest.raises(PermissionError,match='BLOCKED_AUTH'):acquire_nport_pilot(tmp_path,tmp_path)
    assert not (tmp_path/'raw').exists()

def test_nport_bulk_uses_dedicated_validator_and_512mib_budget(tmp_path,monkeypatch):
    import signalforge.sec_pilot as module
    monkeypatch.setenv('SEC_USER_AGENT','SignalForge-QX contact@organization.org')
    seen={}
    class Client:
        def __init__(self,runtime,max_bytes):seen['max_bytes']=max_bytes
        def html(self,url):return '<html></html>',{'path':'discovery'}
        def cached(self,url,kind):seen['cached_kind']=kind;return None
        def fetch(self,url,kind,metadata=None):
            seen['fetch_kind']=kind;seen['metadata']=metadata
            return {'path':str(tmp_path/'nport.zip'),'sha256':'a'*64,'bytes':460_000_000}
    monkeypatch.setattr(module,'Acquisition',Client)
    monkeypatch.setattr(module,'discover_zip',lambda html,source,quarter:'https://www.sec.gov/files/nport-2022q1.zip')
    result=module.acquire_nport_pilot(tmp_path,tmp_path/'runtime')
    assert seen['max_bytes']==512*1024**2
    assert seen['cached_kind']==seen['fetch_kind']=='nport_zip'
    assert result['state']=='SUCCEEDED_RAW_DISSEMINATION_BLOCKED'
    assert result['dissemination']['state']=='BLOCKED_DATA'

def test_sec_syntax_only_not_authentication_and_separate_source_status():
    assert validate_sec_user_agent('Declared Project contact@organization.org')
    text=(REPO/'scripts/acquire_keyed_sources.py').read_text()
    assert all("'"+source+"'" in text for source in ['fred','tiingo','prices','nport'])
    assert 'fred_plan_sha256' in text and 'Existing FRED plan changed' in text

def test_registered_fred_plan_hash_preserved():
    from signalforge.source_requests import validate_fred_plan
    path=REPO/'.local/fred_request_plan.json'
    if not path.exists():pytest.skip('Local registered FRED plan is not distributed with repository')
    before=file_hash(path);plan=json.loads(path.read_text(encoding='utf-8-sig'))
    validate_fred_plan(plan)
    assert file_hash(path)==before
    assert before=='d144daec920dcec77a9fdf51170d468de92c281c092b83f8d43cebc1c8c0ffbb'


def test_missing_origin_features_do_not_reuse_stale_values(tmp_path):
    path=tmp_path/'raw.json';atomic_json(path,[row('2022-04-07'),row('2022-04-21')])
    records=parse_tiingo(path,'SPY')
    origins=['2022-04-08T18:00:00-04:00','2022-04-15T18:00:00-04:00','2022-04-22T18:00:00-04:00']
    marks,events,p1=canonical_inputs(records,origins)
    missing=[e for e in events if e['reference_time']=='2022-04-14T20:00:00+00:00']
    assert len(missing)==4 and all(e['value'] is None for e in missing)
    assert len(marks)==2 and not p1['economic_qualified']


@pytest.mark.parametrize('track',['Main-A','Nested-B'])
def test_i0_partial_grid_preserves_holidays_and_absent_sources(tmp_path,track):
    from signalforge.track_inputs import prepare_track_inputs
    path=tmp_path/'raw.json';atomic_json(path,[row('2022-04-07'),row('2022-04-14'),row('2022-04-21')])
    records=parse_tiingo(path,'SPY')
    origins=['2022-04-08T18:00:00-04:00','2022-04-15T18:00:00-04:00','2022-04-22T18:00:00-04:00']
    marks,events,p1=canonical_inputs(records,origins)
    def card(name,value):
        path=tmp_path/(name+'.json');atomic_json(path,value);return {'relative_path':path.name,'sha256':file_hash(path)}
    manifest={'schema':'release_aware_track_inputs_v1','development_end':END,'reserved_access':False,
        'evidence_kind':'public_data_reconstructed','qualified_for_final':False,'economic_qualified':False,
        'decisions':card('origins',origins),'inputs':{'prices':card('marks',marks),'events':card('events',events)},
        'features':[{'name':'volume','source':'market','field':'log_volume','entity':'ASSET','unit':'log1p_shares'}],
        'universe_card':{'assets':SYMBOLS,'outcome_blind':True,'frozen_before_outcomes':'fixture only'},
        'information_sets':['I0','I1','I2','I3']+(['I4'] if track=='Nested-B' else []),
        'price_mode':'P0','p0_basis':'unadjusted_close_price_return'}
    panels,contexts,q=prepare_track_inputs(REPO,tmp_path,manifest,track)
    assert q['ready_information_sets']==['I0'] and q['grid_rows']==24
    assert len(q['blocked_information_sets'])==(4 if track=='Nested-B' else 3)
    assert not q['qualified_for_final'] and not q['economic_qualified']
    assert len(panels['I0'])==24 and len(contexts['I0'])==24
