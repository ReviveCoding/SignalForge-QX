"""Actual retrieved provenance and additional fail-closed boundary regression tests."""
import json,csv,copy
from pathlib import Path
from datetime import datetime,timedelta,timezone
import pytest
from data_recovery_v36.acquire import R,O,load
from data_recovery_v36.runner import stocks_csv
from data_recovery_v36.html_recovery import Tables
from data_recovery_v36.shadow import validate_message
from data_recovery_v36.cohorts import completeness
from qualification_v35.core import canonical,sha,qualify,signed_record

def test_runner_import_regression():
    from data_recovery_v36.runner import OT
    assert OT.name=='da768f7446b1'
@pytest.mark.parametrize('number',[7,10,13,16])
def test_actual_eia_arithmetic_not_silently_accepted(number):
    d=next(x for x in load(O/'attempted_public_sources.json')['records'] if x['candidate_number']==number)
    rows=stocks_csv(Path(d['payload_path']).read_bytes(),d['sha256'],str(number));assert len(rows)==5
    bad=[x['field'] for x in rows if not x['arithmetic_pass']]
    assert set(bad)==({'commercial_crude','gasoline','distillate'} if number==16 else set())
@pytest.mark.parametrize('number',[24,25])
def test_actual_html_recovery_tables(number):
    d=next(x for x in load(O/'attempted_public_sources.json')['records'] if x['candidate_number']==number);p=Tables();p.feed(Path(d['payload_path']).read_text());assert p.depth==0 and len(p.tables)>=2;assert any('Total' in str(x) for x in p.tables)
@pytest.mark.parametrize('number',[28,29])
def test_actual_cash_never_first_public(number):
    from data_recovery_v36.core import cash_rows
    d=next(x for x in load(O/'attempted_public_sources.json')['records'] if x['candidate_number']==number);rows=cash_rows(Path(d['payload_path']).read_bytes(),d['sha256']);assert len(rows)>=249;assert all(not x['first_public_qualified'] and x['reference_date'].startswith('2023-') for x in rows)
@pytest.mark.parametrize('fault',['payload','publisher','clock','revision','signature'])
def test_actual_document_faults_fail_closed(fault):
    e=copy.deepcopy(load(O/'evidence_inbox.json')['claims'][0]);d=next(x for x in load(O/'attempted_public_sources.json')['records'] if x.get('sha256')==e['payload_sha256']);payload=Path(d['payload_path']).read_bytes()
    if fault=='payload':payload+=b'corrupt'
    elif fault=='publisher':e['publisher']='untrusted.invalid';e['clock_basis']='authenticated_first_public'
    elif fault=='clock':e['known_at']='2023-01-01T12:00:00'
    elif fault=='revision':e['correction_history_complete']=False;e['clock_basis']='authenticated_first_public'
    else:e['dissemination_record']={'key_id':'self-declared','body':{},'signature_b64':'AAA='};e['clock_basis']='authenticated_first_public'
    assert qualify(e,payload,None)['state']!='QUALIFIED_TIER_A'
@pytest.mark.parametrize('key',['unknown','self_enrolled'])
def test_signature_registry_empty(key):assert not signed_record({'key_id':key,'body':{'domain':'v36-pre2024-export-1'},'signature_b64':'AA=='},{},'v36-pre2024-export-1')

def full_fixture():
    p=load(O/'shadow_prepared_plan.json');base=load(O/'shadow_fixture_input.json');rows=[]
    for track in ['Main-A','Nested-B']:
        for b in [0,1]:
            for j in range(52):
                d=datetime(2021+b,1,1,tzinfo=timezone.utc)+timedelta(days=7*j)
                for asset in p['assets']:
                    for seed in p['seeds']:
                        m={**base,'track':track,'block':b,'asset':asset,'seed':seed,'retry_key':f'{track}-{b}-{j}-{asset}-{seed}','decision_at':d.isoformat(),'source_valid_at':(d-timedelta(days=2)).isoformat(),'source_known_at':(d-timedelta(days=1)).isoformat(),'expiry_at':(d+timedelta(hours=1)).isoformat(),'label_end':(d+timedelta(days=7)).isoformat(),'label_available_at':(d+timedelta(days=8)).isoformat()};rows.append(m)
    return rows,p

def test_full_104_date_eight_asset_three_seed_two_track_fixture():
    rows,p=full_fixture();v=completeness(rows,p);assert v['rows']==4992 and v['future_evidence_dates']==0 and v['per_track_dates']=={'Main-A':[52,52],'Nested-B':[52,52]}
@pytest.mark.parametrize('fault',['missing_asset','missing_date','duplicate','real','overlap'])
def test_full_grid_incomplete_or_real_rejected(fault):
    rows,p=full_fixture()
    if fault=='missing_asset':rows.pop()
    elif fault=='missing_date':rows=[x for x in rows if x['decision_at']!=rows[0]['decision_at']]
    elif fault=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif fault=='real':rows[0]['kind']='REAL_FORWARD'
    else:rows[-1]['block']=0
    with pytest.raises((ValueError,PermissionError)):completeness(rows,p)

def test_track_binding_cannot_mix():
    p=load(O/'shadow_prepared_plan.json');m=load(O/'shadow_fixture_input.json');p['track_bindings']={'Main-A':{k+'s':[m[k]] for k in ['model_hash','source_hash','normalizer_hash','target_hash','benchmark_hash']}}
    m['normalizer_hash']=p['normalizer_hashs'][1]
    with pytest.raises(ValueError,match='Track-specific'):validate_message(m,p)
