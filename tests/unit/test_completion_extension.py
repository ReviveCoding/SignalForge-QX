import json
from pathlib import Path
from signalforge.completion import load_protocol, calibration_boundary
from signalforge.tiingo import reconstructed_eod_available_at

ROOT=Path(__file__).resolve().parents[2]

def load(name):
    return json.loads((ROOT/name).read_text())

def test_canonical_successor_protocol_preserves_parent_budget():
    cfg=load_protocol(ROOT)
    assert cfg['protocol_id']=='sgqx-v3.1-track-gpu'
    assert cfg['parent_budget']=={
        'pilot_completed_fit_ceiling':120,
        'development_gpu_seconds':43200,
        'reset_permitted':False,
    }
    assert cfg['successor_gpu']['pilot_fit_ceiling']==60
    assert cfg['successor_gpu']['pilot_gpu_seconds']==3600
    assert cfg['successor_gpu']['full_gpu_seconds']==21600
    assert cfg['successor_gpu']['parent_ledger_reuse'] is False

def test_tiingo_same_friday_final_close_not_available_at_1800():
    import pandas as pd
    close=pd.Timestamp('2023-06-16T16:00:00',tz='America/New_York')
    available=reconstructed_eod_available_at(close)
    decision=pd.Timestamp('2023-06-16T18:00:00',tz='America/New_York').tz_convert('UTC')
    assert available>decision
    assert available.tz_convert('America/New_York').strftime('%H:%M')=='20:30'

def test_calibration_track_boundaries_are_independent(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'reports').mkdir()
    (tmp_path/'configs/statistical_contract.json').write_bytes((ROOT/'configs/statistical_contract.json').read_bytes())
    auxiliary=calibration_boundary(tmp_path,'Auxiliary-C')
    main=calibration_boundary(tmp_path,'Main-A')
    nested=calibration_boundary(tmp_path,'Nested-B')
    assert auxiliary['qualified_for_freeze'] is False
    assert main['auxiliary_block_independent'] is True
    assert nested['auxiliary_block_independent'] is True
    assert main['state']=='NOT_FIT'
    assert nested['state']=='NOT_FIT'

def test_preregistered_source_mapping_is_outcome_blind_and_not_flow():
    cfg=load('configs/track_source_mapping_v31.json')
    assert cfg['outcome_blind'] is True
    assert cfg['rules']['cftc_semantics']=='reported_positioning_contracts_not_cash_flow'
    assert cfg['rules']['unmatched_source']=='missing_not_zero'
    assert set(cfg['targets'])=={'SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'}
    assert cfg['targets']['USO']['eia']
    assert cfg['targets']['SPY']['eia']==[]


def test_successor_track_admission_is_independent_of_nested_nport(tmp_path):
    from signalforge.completion import successor_track_boundary,successor_boundary
    (tmp_path/'configs').mkdir();(tmp_path/'.local').mkdir();(tmp_path/'reports').mkdir()
    (tmp_path/'configs/completion_extension_v31.json').write_text((ROOT/'configs/completion_extension_v31.json').read_text())
    (tmp_path/'configs/local_rtx4090_laptop.json').write_text((ROOT/'configs/local_rtx4090_laptop.json').read_text())
    for track in ['Main-A','Nested-B']:
        (tmp_path/'.local'/f'inputs_{track}_v311.json').write_text('{}')
    def row(track,ready,blocked):
        return {'track':track,'ready_information_sets':ready,'blocked_information_sets':blocked,
            'cpu_development':{'state':'SUCCEEDED_DEVELOPMENT_SOFTWARE','blocked_folds':[],
                'results':[{'information':info,'state':'SUCCEEDED'} for info in ready]}}
    q={'tracks':[
        row('Main-A',['I0','I1','I2','I3'],[]),
        row('Nested-B',['I0','I1','I2','I3'],[{'information':'I4','state':'BLOCKED_DATA','missing_sources':['nport']}]),
    ]}
    (tmp_path/'reports/track_input_qualification_v311.json').write_text(json.dumps(q))
    assert successor_track_boundary(tmp_path,'Main-A')['state']=='READY_FOR_MEASURED_SUCCESSOR_PILOT'
    assert successor_track_boundary(tmp_path,'Nested-B')['state']=='BLOCKED_DATA'
    aggregate=successor_boundary(tmp_path)
    assert aggregate['state']=='PARTIAL_TRACK_READY' and aggregate['ready_tracks']==['Main-A']


def test_gpu_validation_binds_all_suites_and_qualification_content(tmp_path):
    import os
    from signalforge.runtime import code_hash,file_hash,atomic_json
    from signalforge.completion import current_full_validation
    (tmp_path/'reports').mkdir();(tmp_path/'src').mkdir();(tmp_path/'scripts').mkdir()
    q=tmp_path/'reports/track_input_qualification_v311.json';atomic_json(q,{'tracks':[]})
    p=tmp_path/'reports/test_execution.json'
    rows=[{'suite':s,'exit_code':0,'counts':{'tests':1},'source_tree_hash':code_hash(tmp_path)}
          for s in ['handoff','unit','gpu','real_source']]
    doc={'results':rows,'qualification_sha256':file_hash(q),'inputs_unchanged':True}
    atomic_json(p,doc);assert current_full_validation(tmp_path)['passed']
    before=q.stat().st_mtime_ns
    atomic_json(q,{'tracks':[{'changed':True}]});os.utime(q,ns=(before,before))
    assert not current_full_validation(tmp_path)['passed'] # Content binding beats a preserved older mtime.
    doc['qualification_sha256']=file_hash(q);doc['results']=rows[:-1]
    atomic_json(p,doc);assert not current_full_validation(tmp_path)['passed']
    doc['results']=rows;rows[2]['counts']['skipped']=1
    atomic_json(p,doc);assert not current_full_validation(tmp_path)['passed']
    rows[2]['counts']['skipped']=0;(tmp_path/'src/change.py').write_text('changed=True')
    atomic_json(p,doc);assert not current_full_validation(tmp_path)['passed']
