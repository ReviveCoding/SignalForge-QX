"""Synthetic orchestration only: scientific qualification is explicitly stubbed."""
import json
from contextlib import nullcontext
import numpy as np
import pandas as pd
import pytest
from signalforge.runtime import atomic_json,digest,validate_bundle
from signalforge.frozen_processing import candidate_ensemble_id


def setup_batch(tmp_path,monkeypatch,missing_candidate=False):
    from signalforge import final
    repo=tmp_path/'repo';runtime=tmp_path/'runtime';repo.mkdir();runtime.mkdir()
    candidates=[{'candidate_id':name,'track':'Main-A','members':[{'member_id':'11','relative_bundle':'fixture/'+name,'receipt_id':'a'*64}],'weights':[1.]} for name in ['base','challenger']]
    grid=[{'decision_time':d.isoformat(),'asset':'SPY'} for d in pd.date_range('2024-01-05T23:00Z',periods=8,freq='7D')]
    ensemble={'candidate_ensemble_ids':{c['candidate_id']:candidate_ensemble_id(c) for c in candidates}}
    calibration={'candidates':{c['candidate_id']:{'ensemble_id':candidate_ensemble_id(c),'corrections':[0.]*5,'quantiles':[.05,.1,.5,.9,.95],
        'fit_window':{'start':'2023-07-01T00:00Z','end':'2023-12-31T23:59Z','max_label_available_at':'2023-12-30T00:00Z'},'fit_data_id':'b'*64,
        'support':{str(q):{'n_dates':0,'status':'IDENTITY_INSUFFICIENT_SUPPORT'} for q in [.05,.1,.5,.9,.95]}} for c in candidates}}
    features=runtime/'features.json';labels=runtime/'labels.json';inputs=runtime/'inputs.json'
    atomic_json(features,{'tracks':{'Main-A':{'rows':[{**r,'max_dependency_available_at':r['decision_time'],'x':[1.]} for r in grid]}}})
    atomic_json(labels,{'tracks':{'Main-A':{'rows':[{**r,'y':0.,'label_end':(pd.Timestamp(r['decision_time'])+pd.Timedelta(days=7)).isoformat(),'label_available_at':(pd.Timestamp(r['decision_time'])+pd.Timedelta(days=7)).isoformat()} for r in grid]}}})
    atomic_json(inputs,{'features_path':str(features),'labels_path':str(labels)})
    components={}
    contents={'models':{'candidates':candidates,'fallback_candidate_ids':{'Main-A':'base'}},'ensemble':ensemble,'calibration':calibration,
        'contrast_family':{'contrasts':[{'id':'contrast','left':'base','right':'challenger','confirmatory':True}]},
        'features':{'reserved_inputs_manifest':str(inputs),'grid_ids':{'Main-A':digest(grid)}},'metrics_slices':{'scales':{'Main-A':{'SPY':1.}}}}
    for name,content in contents.items():
        path=repo/(name+'.json');atomic_json(path,content);components[name]={'path':str(path)}
    atomic_json(repo/'configs/statistical_contract.json',{'calibration':{'minimum_distinct_dates':{'0.05':52,'0.1':39,'0.5':20,'0.9':39,'0.95':52}}})
    atomic_json(repo/'configs/comparisons.json',{'confirmatory_families':[{'id':'contrast'}]})
    atomic_json(repo/'.local/scientific_authorization.json',{'test_scope':'synthetic; not scientific authorization'})
    receipt={'components':components,'registered_candidate_ids':['base','challenger']+(['absent'] if missing_candidate else []),
        'registered_contrast_ids':['contrast'],'economics_state':'BLOCKED_PRICE_P1'}
    if missing_candidate:
        contents['models']['candidates'].append({**candidates[1],'candidate_id':'absent','track':'Nested-B'})
        atomic_json(repo/'models.json',contents['models'])
        absent=contents['models']['candidates'][-1];identity=candidate_ensemble_id(absent)
        ensemble['candidate_ensemble_ids']['absent']=identity
        calibration['candidates']['absent']={**calibration['candidates']['base'],'ensemble_id':identity}
        atomic_json(repo/'ensemble.json',ensemble);atomic_json(repo/'calibration.json',calibration)
    receipt_path=repo/'receipt.json';atomic_json(receipt_path,receipt)
    freeze='fixture-freeze'
    monkeypatch.setattr('signalforge.integrity.verify_freeze',lambda *a:freeze)
    def admit(*args):
        atomic_json(runtime/'ledger/final_access.json',{'freeze_id':freeze});return freeze
    monkeypatch.setattr(final,'authorize_final',admit)
    monkeypatch.setattr(final,'gpu_lease',lambda *a:nullcontext())
    def inference(candidates,x,*args,**kwargs):
        if candidates[0]['candidate_id']=='challenger':raise ValueError('Intentional complete-grid model failure')
        return {c['candidate_id']:(np.zeros(len(x)),np.tile([-2.,-1.,0.,1.,2.],(len(x),1))) for c in candidates}
    monkeypatch.setattr(final,'infer_candidates',inference)
    return repo,runtime,receipt_path,labels


def test_all_frozen_candidates_commit_before_labels_and_resume_immutable(tmp_path,monkeypatch):
    from signalforge.final import execute_frozen,read_reserved
    repo,runtime,receipt_path,labels=setup_batch(tmp_path,monkeypatch)
    result=execute_frozen(repo,runtime,receipt_path)
    events=[json.loads(s) for s in (runtime/'ledger/final_access_events.jsonl').read_text().splitlines()]
    label_begin=next(i for i,e in enumerate(events) if e.get('kind')=='labels' and e['event']=='READ_BEGIN')
    assert {e.get('candidate_id') for e in events[:label_begin] if e['event']=='PREDICTIONS_COMMITTED'}=={'base','challenger'}
    assert set(result['scores'])=={'base','challenger'} and result['scores']['challenger']['native_prediction_coverage']==0
    assert result['no_model_seed_threshold_selection'] and result['economics_state']=='BLOCKED_PRICE_P1'
    final_directory=runtime/'artifacts/final/fixture-freeze';receipt=validate_bundle(final_directory)
    count=len(events)
    assert execute_frozen(repo,runtime,receipt_path)==result
    replay=[json.loads(s) for s in (runtime/'ledger/final_access_events.jsonl').read_text().splitlines()][count:]
    assert not any(e['event']=='READ_BEGIN' for e in replay)
    payload=json.loads(labels.read_text());payload['tracks']['Main-A']['rows'][0]['y']=100;atomic_json(labels,payload)
    with pytest.raises(PermissionError,match='snapshot changed'):read_reserved(labels,'labels','fixture-freeze',repo,runtime)
    assert validate_bundle(final_directory)==receipt


def test_missing_frozen_track_candidate_prevents_any_label_read(tmp_path,monkeypatch):
    from signalforge.final import execute_frozen
    repo,runtime,receipt_path,_=setup_batch(tmp_path,monkeypatch,True)
    with pytest.raises(PermissionError,match='predictions incomplete'):execute_frozen(repo,runtime,receipt_path)
    events=[json.loads(s) for s in (runtime/'ledger/final_access_events.jsonl').read_text().splitlines()]
    assert not any(e.get('kind')=='labels' for e in events)
