import copy,json
from pathlib import Path
import pytest
from signalforge.source_requests import validate_fred_plan,decode_fred_page,acquire_fred_plan,keyed_pilot
from signalforge.runtime import file_hash

def plan():
    return {'source':'fred','partition':'development','reserved_access':False,'registered_at':'2026-10-01T00:00Z',
        'request_selection':'outcome_blind','max_bytes':1024**2,'max_pages_per_series':3,'page_size':2,
        'series':[{'series_id':'CPIAUCSL','unit':'index_declared_unqualified','native_frequency':'M',
            'observation_start':'2010-01-01','observation_end':'2010-03-31','realtime_start':'2010-06-01','realtime_end':'2010-06-01','output_type':1}]}

def raw(tmp_path,body,name='page'):
    path=tmp_path/(name+'.json');path.write_text(json.dumps(body))
    return {'path':str(path),'sha256':file_hash(path),'source_url':'https://api.stlouisfed.org/fred/series/observations?api_key=REDACTED'}

def body(offset,rows,count=3,output=1):
    return {'offset':offset,'limit':2,'count':count,'units':'lin','output_type':output,'observations':rows,
        'realtime_start':'2010-06-01' if output==1 else '2010-01-01','realtime_end':'2010-06-01' if output==1 else '2010-12-31'}

def observation(day):return {'date':day,'realtime_start':'2010-06-01','realtime_end':'2010-06-01','value':'1.5'}

def test_fred_plan_rejects_latest_reserved_secret_and_unbounded_defaults():
    validate_fred_plan(plan())
    for change in [{'reserved_access':True},{'max_bytes':2*1024**3},{'request_selection':'selected_from_outcomes'},{'api_key':'synthetic-secret'}]:
        with pytest.raises((PermissionError,ValueError)):validate_fred_plan({**plan(),**change})
    for field,value in [('observation_end','2024-01-01'),('realtime_end','2024-01-01'),('output_type',4)]:
        changed=plan();changed['series'][0][field]=value
        with pytest.raises((PermissionError,ValueError)):validate_fred_plan(changed)

def test_fred_snapshot_pagination_preserves_asof_not_first_publication(tmp_path,monkeypatch):
    monkeypatch.setenv('FRED_API_KEY','a'*32)
    pages=[raw(tmp_path,body(0,[observation('2010-01-01'),observation('2010-02-01')]),'zero'),
        raw(tmp_path,body(2,[observation('2010-03-01')]),'two')]
    class Client:
        def cached(self,url,kind,params):return None
        def fetch(self,url,kind,params,metadata):
            assert params['api_key']=='a'*32 and params['offset'] in {0,2}
            assert params['realtime_start']==params['realtime_end']=='2010-06-01'
            return pages[params['offset']//2]
    result=acquire_fred_plan(tmp_path,tmp_path/'runtime',plan(),Client())
    assert len(result['records'])==3 and result['pit_tier']=='B' and not result['model_qualified']
    assert result['records'][0]['reference_time']=='2010-01-31T00:00:00+00:00'
    assert all(r['available_at']=='2010-06-02T12:00:00+00:00' and 'not_first_publication' in r['clock_evidence'] for r in result['records'])
    assert 'a'*32 not in json.dumps(result)

def test_fred_truncated_changed_count_or_wrong_vintage_cannot_complete(tmp_path):
    s=plan()['series'][0]
    for payload in [body(0,[observation('2010-01-01')]),body(1,[observation('2010-01-01'),observation('2010-02-01')]),
        body(0,[{**observation('2010-01-01'),'realtime_start':'2026-01-01'},observation('2010-02-01')])]:
        with pytest.raises((ValueError,PermissionError)):decode_fred_page(raw(tmp_path,payload),s,0,2)
    with pytest.raises(ValueError,match='count changed'):decode_fred_page(raw(tmp_path,body(0,[observation('2010-01-01'),observation('2010-02-01')])),s,0,2,4)

def test_fred_revision_columns_keep_date_only_vintage_and_missingness(tmp_path):
    s=plan()['series'][0];s.update(output_type=3,realtime_start='2010-01-01',realtime_end='2010-12-31')
    rows=[{'date':'2010-01-01','CPIAUCSL_20100201':'.','CPIAUCSL_20100301':'2.5'}]
    parsed,count,progress=decode_fred_page(raw(tmp_path,body(0,rows,1,3)),s,0,2)
    assert count==progress==1 and len(parsed)==2 and parsed[0]['value'] is None
    assert parsed[1]['value']==2.5 and parsed[1]['realtime_end'] is None and not parsed[1]['original_publication_qualified']
    rows[0]['CPIAUCSL_20240101']='999'
    with pytest.raises(PermissionError):decode_fred_page(raw(tmp_path,body(0,rows,1,3)),s,0,2)

def test_fred_requires_key_before_any_network_and_actual_pages_before_success(tmp_path,monkeypatch):
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    with pytest.raises(PermissionError,match='absent'):acquire_fred_plan(tmp_path,tmp_path/'runtime',plan())
    monkeypatch.setenv('FRED_API_KEY','synthetic-key')
    with pytest.raises(PermissionError,match='32 lowercase'):acquire_fred_plan(tmp_path,tmp_path/'runtime',plan())
    monkeypatch.setenv('FRED_API_KEY','a'*32)
    class Empty:
        def cached(self,*args):return raw(tmp_path,body(0,[],0))
    with pytest.raises(RuntimeError,match='no observation versions'):acquire_fred_plan(tmp_path,tmp_path/'runtime',plan(),Empty())

def test_weekly_prices_cannot_download_reserved_history_or_premium_daily(tmp_path,monkeypatch):
    monkeypatch.setenv('ALPHAVANTAGE_API_KEY','synthetic-key')
    result=keyed_pilot(tmp_path,tmp_path/'runtime','prices')
    assert result['state']=='BLOCKED_SOURCE_TIME_SCOPE' and not result['reserved_access'] and not result['model_qualified']
    assert not (tmp_path/'runtime/raw').exists()

def test_fred_initial_snapshot_and_revision_segments_cannot_overlap_or_change_units():
    p=plan();revision={**p['series'][0],'output_type':3,'realtime_start':'2010-06-02','realtime_end':'2010-12-31'}
    p['series'].append(revision);validate_fred_plan(p)
    revision['realtime_start']='2010-06-01'
    with pytest.raises(ValueError,match='disjoint'):validate_fred_plan(p)
    revision['realtime_start']='2010-06-02';revision['unit']='silently_changed'
    with pytest.raises(ValueError,match='unit/frequency'):validate_fred_plan(p)


def test_fred_segmented_revision_plan_has_conservative_vintage_bound():
    import pandas as pd
    from signalforge.source_requests import FRED_SEGMENT_MAX_CALENDAR_DAYS
    p=plan()
    base=p['series'][0]
    p['series']=[
        base,
        {**base,'output_type':3,'realtime_start':'2010-06-02','realtime_end':'2010-12-31'},
        {**base,'output_type':3,'realtime_start':'2011-01-01','realtime_end':'2011-12-31'},
    ]
    validate_fred_plan(p,require_json_safe_segments=True)
    bad=copy.deepcopy(p)
    bad['series']=bad['series'][:2]
    bad['series'][1]['realtime_end']=str((pd.Timestamp('2010-06-02')+pd.Timedelta(days=FRED_SEGMENT_MAX_CALENDAR_DAYS)).date())
    with pytest.raises(ValueError,match='safety bound'):
        validate_fred_plan(bad,require_json_safe_segments=True)


def cached_dgs10(offset):
    path=Path(__file__).resolve().parents[1]/'fixtures/fred'/f'dgs10_revision_offset_{offset}.json'
    checksums={0:'2863fc9b22e6c4a481f3a19614c0743a7431f591fd93bff578c523d4f6ee9102',
               1000:'90b58923909a645772aeff63cc644f9d64da96ab363f7997d602546f7bf0e109'}
    return {'path':str(path),'sha256':checksums[offset]}


def dgs10_revision():
    return {'series_id':'DGS10','unit':'Percent','native_frequency':'D',
            'observation_start':'2000-01-01','observation_end':'2023-12-31',
            'realtime_start':'2010-07-21','realtime_end':'2015-05-05','output_type':3}


def test_actual_sparse_dgs10_page_is_not_truncated():
    values,count,progress=decode_fred_page(cached_dgs10(1000),dgs10_revision(),1000,1000,4002)
    assert count==4002 and len(values)==251 and progress==251
    assert values[0]['reference_label']=='2014-05-19'
    assert values[-1]['reference_label']=='2015-05-04'


@pytest.mark.parametrize('continuation_offset',[1251,2251])
def test_sparse_acquisition_covers_every_offset_window_even_after_short_or_empty_page(tmp_path,monkeypatch,continuation_offset):
    monkeypatch.setenv('FRED_API_KEY','a'*32) # Mock client only; never an actual provider request.
    p=plan();p.update(page_size=1000,max_pages_per_series=6)
    revision=dgs10_revision()
    snapshot={**revision,'output_type':1,'realtime_start':'2010-07-20','realtime_end':'2010-07-20'}
    p['series']=[snapshot,revision]
    visited=[]
    class Client:
        def cached(self,url,kind,params):
            if params['output_type']==1:
                b={**body(0,[{**observation('2000-01-03'),'realtime_start':'2010-07-20','realtime_end':'2010-07-20'}],1),
                   'limit':1000,'realtime_start':'2010-07-20','realtime_end':'2010-07-20'}
                return raw(tmp_path,b,'snapshot')
            offset=params['offset'];visited.append(offset)
            if offset in {0,1000}:return cached_dgs10(offset)
            b=json.loads(Path(cached_dgs10(1000)['path']).read_text())
            b.update(offset=offset,observations=[])
            # A later populated page proves short/empty pages do not silently terminate acquisition.
            if offset==continuation_offset:b['observations']=[{'date':'2015-05-05','DGS10_20150505':'2.17'}]
            return raw(tmp_path,b,str(offset))
    result=acquire_fred_plan(tmp_path,tmp_path/'runtime',p,Client())
    assert visited==([0,1000,1251,1252,2252,3252] if continuation_offset==1251 else [0,1000,1251,2251,2252,3252])
    assert len(result['records'])==1254 # 1 snapshot + 1001 cells + 251 cells + 1 later cell.
    assert result['pagination_segments'][1]['covered_offset_end']==4002
    assert [x['observation_rows'] for x in result['pagination_segments'][1]['pages']]==([1000,251,1,0,0,0] if continuation_offset==1251 else [1000,251,0,1,0,0])


@pytest.mark.parametrize('change',[
    {'offset':999},{'count':4003},{'units':'pch'},{'output_type':1},
    {'realtime_start':'2010-07-20'},{'realtime_end':'2024-01-01'},
])
def test_sparse_page_metadata_still_fails_closed(tmp_path,change):
    b=json.loads(Path(cached_dgs10(1000)['path']).read_text());b.update(change)
    with pytest.raises((ValueError,PermissionError)):
        decode_fred_page(raw(tmp_path,b),dgs10_revision(),1000,1000,4002)


@pytest.mark.parametrize('row',[
    {'date':'2024-01-01','DGS10_20150505':'2'},
    {'date':'2015-05-05','DGS10_20240101':'2'},
    {'date':'2015-05-05','DGS10_20150505':'nan'},
    {'date':'2015-05-05','DGS10_20150505':'not numeric'},
])
def test_sparse_values_still_fail_closed(tmp_path,row):
    b=json.loads(Path(cached_dgs10(1000)['path']).read_text());b['observations']=[row]
    with pytest.raises((ValueError,PermissionError)):
        decode_fred_page(raw(tmp_path,b),dgs10_revision(),1000,1000,4002)


def test_sparse_page_corruption_duplicate_cells_and_dates_rejected(tmp_path):
    receipt=cached_dgs10(1000)
    with pytest.raises(ValueError,match='raw source changed'):
        decode_fred_page({**receipt,'sha256':'0'*64},dgs10_revision(),1000,1000,4002)
    b=json.loads(Path(receipt['path']).read_text());b['observations']*=2
    with pytest.raises(ValueError,match='ascending'):
        decode_fred_page(raw(tmp_path,b),dgs10_revision(),1000,1000,4002)
    path=tmp_path/'duplicate.json'
    path.write_text(Path(receipt['path']).read_text().replace('"DGS10_20140520":"2.54"',
        '"DGS10_20140520":"2.54","DGS10_20140520":"2.55"'))
    # Preserve original fixture whitespace, which is provider-dependent.
    if path.read_text()==Path(receipt['path']).read_text():
        text=json.dumps(json.loads(Path(receipt['path']).read_text()))
        path.write_text(text.replace('"DGS10_20140520": "2.54"','"DGS10_20140520": "2.54", "DGS10_20140520": "2.55"'))
    with pytest.raises(ValueError,match='Duplicate FRED JSON'):
        decode_fred_page({'path':str(path),'sha256':file_hash(path)},dgs10_revision(),1000,1000,4002)
