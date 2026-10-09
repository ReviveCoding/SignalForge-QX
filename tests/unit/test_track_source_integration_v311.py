import json
from pathlib import Path
import pandas as pd

from signalforge.runtime import atomic_json,file_hash
from signalforge.p0_labels import last_regular_close
from signalforge.track_source_integration import build_extension_v311

ROOT=Path(__file__).resolve().parents[2]

def test_v311_source_extension_opens_main_i1_i3_without_nport(tmp_path,monkeypatch):
    repo=tmp_path/'repo';runtime=tmp_path/'runtime'
    (repo/'.local').mkdir(parents=True);(repo/'reports').mkdir();(repo/'configs').mkdir();runtime.mkdir()
    (repo/'configs/study.json').write_text((ROOT/'configs/study.json').read_text())
    (repo/'configs/track_feature_specs_v311.json').write_text((ROOT/'configs/track_feature_specs_v311.json').read_text())

    dates=pd.date_range('2022-01-07',periods=6,freq='W-FRI',tz='America/New_York')+pd.Timedelta(hours=18)
    assets=['SPY']
    def card(name,value):
        p=runtime/(name+'.json');atomic_json(p,value)
        return {'relative_path':p.name,'sha256':file_hash(p)}

    events=[];prices=[]
    for i,d in enumerate(dates):
        close=last_regular_close(d)
        events.append({'entity':'SPY','source':'market','field':'close','reference_time':close.isoformat(),
            'available_at':close.isoformat(),'value':100.+i,'unit':'USD/share','raw_hash':'1'*64,'pit_tier':'B'})
        prices.append({'asset':'SPY','decision_time':d.isoformat(),'time':close.isoformat(),'available_at':close.isoformat(),
            'close':100.+i,'fresh':True,'adjusted':False,'raw_hash':'1'*64})
    base={
        'schema':'release_aware_track_inputs_v1','reserved_access':False,'development_end':'2023-12-31',
        'evidence_kind':'public_data_reconstructed','decisions':card('decisions',[d.isoformat() for d in dates]),
        'universe_card':{'assets':assets,'outcome_blind':True,'frozen_before_outcomes':'fixture'},
        'information_sets':['I0','I1','I2','I3'],'price_mode':'P0','p0_basis':'unadjusted_close_price_return',
        'inputs':{'events':card('market_events',events),'prices':card('prices',prices)},
        'features':[{'name':'market_close','source':'market','field':'close','entity':'ASSET','unit':'USD/share'}],
        'qualified_for_final':False,'economic_qualified':False,
    }
    atomic_json(repo/'.local/inputs_Main-A.json',base)
    nested=json.loads(json.dumps(base));nested['information_sets']=['I0','I1','I2','I3','I4']
    atomic_json(repo/'.local/inputs_Nested-B.json',nested)

    source_events={
        'macro':[{'entity':'CPIAUCSL','source':'macro','field':'value','reference_time':'2021-12-31T00:00:00+00:00',
                  'available_at':'2022-01-01T12:00:00+00:00','value':280.,'unit':'Index 1982-1984=100','raw_hash':'2'*64,'pit_tier':'B'}],
        'cftc':[{'entity':'SPY','source':'cftc','field':'Dealer_net_fraction_open_interest','reference_time':'2021-12-28T00:00:00+00:00',
                 'available_at':'2021-12-31T20:00:00+00:00','value':0.1,'unit':'fraction_open_interest','raw_hash':'3'*64,'pit_tier':'B'}],
        'eia':[{'entity':'USO','source':'eia','field':'commercial_crude_change','reference_time':'2021-12-31T00:00:00+00:00',
                'available_at':'2022-01-05T16:00:00+00:00','value':1.0,'unit':'million_barrels','raw_hash':'4'*64,'pit_tier':'B'}],
    }
    states={
        'macro':('macro_canonical_integration.json','SUCCEEDED_RECONSTRUCTED_MACRO_VINTAGES'),
        'cftc':('cftc_canonical_integration.json','SUCCEEDED_RECONSTRUCTED_CFTC_POSITIONING'),
        'eia':('eia_market_integration.json','SUCCEEDED_RECONSTRUCTED_EIA_USO_ACTIVITY'),
    }
    for source,rows in source_events.items():
        p=runtime/(source+'.json');atomic_json(p,rows)
        name,state=states[source]
        atomic_json(repo/'reports'/name,{'state':state,'events_relative_path':p.name,'events_sha256':file_hash(p)})

    result=build_extension_v311(repo,runtime)
    assert result['available_sources']==['cftc','eia','macro']
    rows={r['track']:r for r in result['tracks']}
    assert rows['Main-A']['ready_information_sets']==['I0','I1','I2','I3']
    assert rows['Main-A']['blocked_information_sets']==[]
    assert rows['Main-A']['base_i0_signature']==rows['Main-A']['successor_i0_signature']
    assert rows['Nested-B']['ready_information_sets']==['I0','I1','I2','I3']
    assert rows['Nested-B']['blocked_information_sets']==[
        {'information':'I4','state':'BLOCKED_DATA','missing_sources':['nport']}
    ]
    assert rows['Nested-B']['base_i0_signature']==rows['Nested-B']['successor_i0_signature']
    # A failed re-build must leave the previously verified manifest intact.
    import pytest
    import signalforge.track_source_integration as integration
    previous=(repo/'.local/inputs_Main-A_v311.json').read_bytes()
    signatures=iter(['base','changed'])
    monkeypatch.setattr(integration,'_i0_signature',lambda *a:next(signatures))
    with pytest.raises(ValueError,match='I0 parity changed'):
        build_extension_v311(repo,runtime)
    assert (repo/'.local/inputs_Main-A_v311.json').read_bytes()==previous
