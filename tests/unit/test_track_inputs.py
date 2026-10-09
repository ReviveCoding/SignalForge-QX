import json
from pathlib import Path
import pytest
from signalforge.track_inputs import read_input_card,prepare_track_inputs,qualify_track_inputs
from signalforge.runtime import file_hash


def test_input_hash_and_ext4_path_scope(tmp_path):
    path=tmp_path/'calendar.json';path.write_text('["2020-01-03T23:00Z"]')
    card={'relative_path':path.name,'sha256':file_hash(path)}
    assert read_input_card(tmp_path,card)==['2020-01-03T23:00Z']
    path.write_text('[]')
    with pytest.raises(ValueError,match='checksum'):read_input_card(tmp_path,card)
    with pytest.raises(PermissionError,match='escapes'):read_input_card(tmp_path,{'relative_path':'../outside.json','sha256':'a'*64})


def test_origin_guard_before_opening_target_files(tmp_path):
    path=tmp_path/'calendar.json';path.write_text('["2024-01-05T23:00Z"]')
    manifest={'schema':'release_aware_track_inputs_v1','reserved_access':False,'development_end':'2023-12-31',
        'evidence_kind':'public_data_reconstructed','decisions':{'relative_path':path.name,'sha256':file_hash(path)},
        'universe_card':{'outcome_blind':True,'frozen_before_outcomes':'registered','assets':['SPY']}}
    # No inputs key: the reserved-origin denial must occur first.
    with pytest.raises(PermissionError,match='reserved'):prepare_track_inputs(tmp_path,tmp_path,manifest,'Main-A')
    manifest['evidence_kind']='synthetic_fixture'
    with pytest.raises(PermissionError,match='Fixture'):prepare_track_inputs(tmp_path,tmp_path,manifest,'Main-A')


def test_absent_public_input_manifest_preserves_independent_blockers(tmp_path):
    for track in ['Main-A','Nested-B']:
        panels,contexts,result=qualify_track_inputs(tmp_path,tmp_path,track)
        assert panels is contexts is None and result['state']=='BLOCKED_DATA'
        assert not result['reserved_access'] and not result['qualified_for_final']


def test_available_price_information_branch_continues_when_flow_sources_absent(tmp_path):
    import pandas as pd
    from signalforge.p0_labels import last_regular_close
    from signalforge.runtime import atomic_json
    repo=Path(__file__).parents[2];dates=pd.date_range('2022-01-07',periods=5,freq='W-FRI',tz='America/New_York')+pd.Timedelta(hours=18)
    def card(name,value):
        path=tmp_path/(name+'.json');atomic_json(path,value);return {'relative_path':path.name,'sha256':file_hash(path)}
    events=[{'entity':'a','source':'market','field':'close','reference_time':last_regular_close(d).isoformat(),
        'available_at':last_regular_close(d).isoformat(),'value':100+i,'unit':'USD/share','raw_hash':'a'*64,'pit_tier':'B'} for i,d in enumerate(dates)]
    prices=[{'asset':'a','decision_time':d.isoformat(),'time':last_regular_close(d).isoformat(),'available_at':last_regular_close(d).isoformat(),
        'close':100+i,'fresh':True,'adjusted':False,'raw_hash':'a'*64} for i,d in enumerate(dates)]
    manifest={'schema':'release_aware_track_inputs_v1','reserved_access':False,'development_end':'2023-12-31','evidence_kind':'public_data_reconstructed',
        'decisions':card('dates',[d.isoformat() for d in dates]),'universe_card':{'assets':['a'],'outcome_blind':True,'frozen_before_outcomes':'fixture only'},
        'information_sets':['I0','I1','I2','I3'],'price_mode':'P0','p0_basis':'unadjusted_close_price_return',
        'inputs':{'events':card('events',events),'prices':card('prices',prices)},
        'features':[{'name':s,'source':s,'field':'close','entity':'ASSET','unit':'USD/share'} for s in ['market','macro','cftc','eia']]}
    panels,contexts,result=prepare_track_inputs(repo,tmp_path,manifest,'Main-A')
    assert list(panels)==['I0'] and result['state']=='PARTIAL_RECONSTRUCTED_DEVELOPMENT_INPUTS'
    assert len(result['blocked_information_sets'])==3 and result['grid_rows']==5
    assert len(contexts['I0'])==5 and not result['qualified_for_final'] and not result['economic_qualified']

def test_partitioned_event_cards_preserve_bounds_and_order(tmp_path):
    from signalforge.track_inputs import read_partitioned_cards
    from signalforge.runtime import atomic_json
    def card(name,rows):
        p=tmp_path/(name+'.json');atomic_json(p,rows)
        return {'relative_path':p.name,'sha256':file_hash(p)}
    a=card('a',[{'x':1},{'x':2}])
    b=card('b',[{'x':3}])
    assert read_partitioned_cards(tmp_path,[a,b])==[{'x':1},{'x':2},{'x':3}]
    with pytest.raises(ValueError,match='Duplicate'):
        read_partitioned_cards(tmp_path,[a,a])

def test_prepare_track_inputs_rejects_mixed_single_and_partitioned_events(tmp_path):
    import pandas as pd
    from signalforge.p0_labels import last_regular_close
    from signalforge.runtime import atomic_json
    repo=Path(__file__).parents[2]
    dates=pd.date_range('2022-01-07',periods=5,freq='W-FRI',tz='America/New_York')+pd.Timedelta(hours=18)
    def card(name,value):
        p=tmp_path/(name+'.json');atomic_json(p,value)
        return {'relative_path':p.name,'sha256':file_hash(p)}
    events=[{'entity':'a','source':'market','field':'close','reference_time':last_regular_close(d).isoformat(),
        'available_at':last_regular_close(d).isoformat(),'value':100+i,'unit':'USD/share','raw_hash':'a'*64,'pit_tier':'B'} for i,d in enumerate(dates)]
    prices=[{'asset':'a','decision_time':d.isoformat(),'time':last_regular_close(d).isoformat(),'available_at':last_regular_close(d).isoformat(),
        'close':100+i,'fresh':True,'adjusted':False,'raw_hash':'a'*64} for i,d in enumerate(dates)]
    ev=card('events',events)
    manifest={'schema':'release_aware_track_inputs_v1','reserved_access':False,'development_end':'2023-12-31',
        'evidence_kind':'public_data_reconstructed','decisions':card('dates',[d.isoformat() for d in dates]),
        'universe_card':{'assets':['a'],'outcome_blind':True,'frozen_before_outcomes':'fixture only'},
        'information_sets':['I0','I1','I2','I3'],'price_mode':'P0','p0_basis':'unadjusted_close_price_return',
        'inputs':{'events':ev,'event_partitions':[ev],'prices':card('prices',prices)},
        'features':[{'name':'market_close','source':'market','field':'close','entity':'ASSET','unit':'USD/share'}]}
    with pytest.raises(ValueError,match='either one card or registered partitions'):
        prepare_track_inputs(repo,tmp_path,manifest,'Main-A')
