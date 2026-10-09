import copy,json
from pathlib import Path
import pytest
from signalforge.runtime import digest
from signalforge.v33_admission import combinations,admission,require_admission,LEDGER,PROTOCOL
def fixture():
    c=json.loads((Path(__file__).parents[2]/'configs/v33_retrospective_diagnostic.json').read_text())
    rows=[dict(r,feasible=True,fold_id='fixture-fold',source_id='fixture-source',sia_manifest_id='fixture-sia',normalizer_id='fixture-normalizer') for r in combinations(c)]
    p={'protocol_id':PROTOCOL,'registered_config':c,'fits':rows,'source_tree_hash':'fixture-code','reserved_access':False};p['plan_id']=digest(p)
    b={'protocol_id':PROTOCOL,'plan_id':p['plan_id'],'measured_pilot':True,'pilot_receipt_sha256':'fixture-not-real-pilot','outcome_blind':True,'ceiling_seconds':1000.,'conservative_projected_seconds':600.,'ledger_path':LEDGER,'parent_ledger_reuse':False,'planned_fits':36,'safety_margin':1.5}
    b['measurement_source_hash']='fixture-code';b['bill_id']=digest(b)
    a={'protocol_id':PROTOCOL,'plan_id':p['plan_id'],'ceiling_seconds':1000.,'ledger_path':LEDGER,'user_cost_authorized':True}
    q={'scope':'v33_execution','passed':True,'gpu_contract_passed':True,'source_tree_hash':'fixture-code','plan_id':p['plan_id']}
    return p,b,a,q
def test_exact_36_no_implicit_grid():
    p,*_=fixture();assert len(p['fits'])==36 and len({(r['track'],r['year'],r['family'],r['seed']) for r in p['fits']})==36
def test_missing_real_numeric_authorization_blocks():
    p,*_=fixture(); result=admission(p)
    assert result['state']=='BLOCKED_RESOURCE_ADMISSION' and not result['training_started']
    with pytest.raises(PermissionError):require_admission(p)
@pytest.mark.parametrize('field,value',[('ceiling_seconds',None),('ceiling_seconds',True),('ceiling_seconds',float('nan')),('conservative_projected_seconds',2000),('ledger_path','/home/USERNAME/original/development_compute.sqlite'),('parent_ledger_reuse',True),('measured_pilot',False),('safety_margin',1.)])
def test_bad_bill(field,value):
    p,b,a,q=fixture();b[field]=value
    assert admission(p,b,a,q)['blocked_reasons']
def test_synthetic_checker_positive_is_not_training():
    p,b,a,q=fixture();r=admission(p,b,a,q)
    assert r['state']=='ELIGIBLE_FOR_LEASE_CHECK_ONLY' and r['training_started'] is False
def test_stale_code_and_auth_mismatch():
    p,b,a,q=fixture();q['source_tree_hash']='wrong';a['ceiling_seconds']=999
    assert len(admission(p,b,a,q)['blocked_reasons'])>=2
def test_corrupted_plan_and_grid_fail():
    p,b,a,q=fixture();p['fits'][0]['seed']=999
    with pytest.raises(PermissionError):admission(p,b,a,q)
def test_unknown_fold_qual_blocks():
    p,b,a,q=fixture();p.pop('plan_id');p['fits'][0]['fold_id']=None;p['plan_id']=digest(p)
    assert 'EXACT_FOLD_SOURCE_TRANSFORM_QUALIFICATION' in admission(p)['blocked_reasons']

@pytest.mark.parametrize('count,passed',[(39,False),(51,False),(52,True)])
def test_post_context_dates_not_pooled_asset_rows(count,passed):
    import pandas as pd
    from signalforge.v33_admission import fold_support
    dates=pd.date_range('2020-01-01',periods=count,freq='7D',tz='UTC')
    frame=pd.DataFrame({'decision_time':list(dates)*8})
    assert fold_support(frame,52)['passed'] is passed
    assert fold_support(frame,52)['independent_mature_context_dates']==count