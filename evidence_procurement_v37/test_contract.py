"""Synthetic signed export/temporal/licensing/action properties. Never scientific data."""
import json,copy,base64,subprocess,sqlite3
from pathlib import Path
from unittest.mock import patch
import pytest
from qualification_v35.core import canonical,sha
from evidence_procurement_v37.core import admission,accept,register_import,validate_row
from evidence_procurement_v37.tiingo_beta_action_bridge import reconcile

@pytest.fixture
def fixture(tmp_path):
    key=tmp_path/'TEST_ONLY.pem';subprocess.run(['openssl','genpkey','-algorithm','ED25519','-out',str(key)],capture_output=True,check=True);pub=subprocess.check_output(['openssl','pkey','-in',str(key),'-pubout','-outform','DER'])[-32:].hex()
    def sign(body):
        (tmp_path/'sign_body').write_bytes(canonical(body));subprocess.run(['openssl','pkeyutl','-sign','-inkey',str(key),'-rawin','-in',str(tmp_path/'sign_body'),'-out',str(tmp_path/'sig')],capture_output=True,check=True);return {'key_id':'TEST_ONLY_CUSTODIAN','body':body,'signature_b64':base64.b64encode((tmp_path/'sig').read_bytes()).decode()}
    row={'event_id':'TEST_ONLY_ACTION','version_id':'v1','provider':'TEST_ONLY_VENDOR','publisher':'fixture.invalid','kind':'dividend','entity':'IEF','field':'distribution','unit':'USD_per_share','valid_at':'2023-02-01T00:00:00Z','known_at':None,'first_public_at':None,'clock_basis':'unverified','correction_chain':[],'values':{'ticker':'IEF','permaTicker':'TEST_ONLY_IEF','exDate':'2023-02-01','paymentDate':'2023-02-07','recordDate':'2023-02-02','declarationDate':'2023-01-20','distribution':.212178,'distributionFrequency':'m','currency':'USD'},'test_not_evidence':True}
    body={'domain':'v37-authorized-pre2024-export-1','dataset_id':'TEST_ONLY_DATASET','provider':'TEST_ONLY_VENDOR','publisher':'fixture.invalid','scope':'PRE_2024_ONLY','snapshot_asof':'2023-12-31T23:59:59Z','range_start':'2023-01-01','range_end':'2023-12-31','min_valid_at':'2023-01-01T00:00:00Z','max_valid_at':'2023-12-31T23:59:59Z','min_known_at':'2023-01-01T00:00:00Z','max_known_at':'2023-12-31T23:59:59Z','assets':['IEF'],'series':[],'row_count':1,'payload_bytes':0,'payload_sha256':'0'*64,'version_ids':['v1'],'excluded_rows_count':1,'independent_pre2024_export':True,'no_opaque_attachments':True,'license':{'research_use_permitted':True,'local_analysis_permitted':True,'entitlement_confirmed':True,'restrictions_resolved':True,'range_start':'2023-01-01','range_end':'2023-12-31','assets':['IEF'],'series':[],'license_record_id':'TEST_ONLY_LICENSE','public_redistribution_permitted':False},'test_not_evidence':True}
    registry={'schema':'v37-trust-1','environment':'TEST_ONLY','independently_enrolled':True,'custodians':{'TEST_ONLY_CUSTODIAN':{'role':'authorized_export_custodian','public_key_hex':pub,'providers':['TEST_ONLY_VENDOR'],'publishers':['fixture.invalid'],'enrollment_evidence_sha256':'a'*64,'test_not_evidence':True}},'publishers':{},'reviewers':{}}
    inbox=tmp_path/'inbox';inbox.mkdir();path=inbox/'TEST_ONLY.json'
    def prepare(rows=None,b=None):
        rows=rows or [copy.deepcopy(row)];b=copy.deepcopy(b or body);payload=canonical({'schema':'v37-authorized-events-1','records':rows});path.write_bytes(payload);b.update(row_count=len(rows),payload_bytes=len(payload),payload_sha256=sha(payload),version_ids=sorted(set(r['version_id'] for r in rows)));return path,sign(b),registry,inbox
    return row,body,registry,sign,prepare,path,inbox

def run(fixture,rows=None,body=None):
    p,m,r,i=fixture[4](rows,body);return accept(p,m,r,i,test_only=True)

def test_signed_fixture_acceptance_is_not_evidence(fixture):
    out=run(fixture);assert out['state']=='ACCEPTED_TEST_ONLY' and out['qualified_Tier_A']==out['P1_qualified_sessions']==0 and not out['real_forward_allowed']
@pytest.mark.parametrize('fault',['mixed_known','mixed_valid','snapshot','range','unsigned','key','provider','license','entitlement','license_period','custodian_role','opaque','scope','not_independent','real_fixture'])
def test_admission_rejection_without_payload_read(fixture,fault):
    row,b,reg,sign,prepare,path,inbox=fixture;p,m,reg,i=prepare();b=copy.deepcopy(m['body']);reg=copy.deepcopy(reg)
    if fault=='mixed_known':b['max_known_at']='2024-01-01T00:00:00Z'
    elif fault=='mixed_valid':b['max_valid_at']='2024-01-01T00:00:00Z'
    elif fault=='snapshot':b['snapshot_asof']='2024-01-01T00:00:00Z'
    elif fault=='range':b['range_end']='2024-01-01'
    elif fault=='provider':b['provider']='NOT_AUTHORIZED'
    elif fault=='license':b['license']['local_analysis_permitted']=False
    elif fault=='entitlement':b['license']['entitlement_confirmed']=False
    elif fault=='license_period':b['range_start']='2010-07-20'
    elif fault=='custodian_role':reg['custodians']['TEST_ONLY_CUSTODIAN']['role']='self_declared'
    elif fault=='opaque':b['no_opaque_attachments']=False
    elif fault=='scope':b['scope']='UNSCOPED_MIXED'
    elif fault=='not_independent':reg['independently_enrolled']=False
    elif fault=='real_fixture':reg['environment']='REAL'
    m=sign(b)
    if fault=='unsigned':m['signature_b64']=base64.b64encode(b'0'*64).decode()
    if fault=='key':m['key_id']='UNKNOWN'
    with patch.object(Path,'read_bytes',side_effect=AssertionError('Forbidden payload touched')):
        with pytest.raises((PermissionError,ValueError)):accept(p,m,reg,i,test_only=True)

@pytest.mark.parametrize('fault',['wrong_sha','extra_field','nonfinite','duplicate','version_roster','unknown_asset','declaration_after','pay_before','record_after','ambiguous_timezone','reserved_row','hidden_extra','negative','bad_frequency'])
def test_payload_semantic_faults(fixture,fault):
    row=copy.deepcopy(fixture[0]);rows=[row]
    if fault=='extra_field':row['new']='unexpected'
    elif fault=='nonfinite':row['values']['distribution']=float('inf')
    elif fault=='duplicate':rows.append(copy.deepcopy(row))
    elif fault=='unknown_asset':row['values']['ticker']='SPY'
    elif fault=='declaration_after':row['values']['declarationDate']='2023-03-01'
    elif fault=='pay_before':row['values']['paymentDate']='2023-01-01'
    elif fault=='record_after':row['values']['recordDate']='2023-02-08'
    elif fault=='ambiguous_timezone':row['known_at']='2023-01-20T00:00:00'
    elif fault=='reserved_row':row['known_at']='2024-01-01T00:00:00Z'
    elif fault=='hidden_extra':row['values']['hidden2024']='synthetic forbidden field'
    elif fault=='negative':row['values']['distribution']=-1
    elif fault=='bad_frequency':row['values']['distributionFrequency']='unknown'
    if fault=='nonfinite':
        with pytest.raises(ValueError):fixture[4](rows)
        return
    p,m,reg,i=fixture[4](rows)
    if fault=='wrong_sha':p.write_bytes(p.read_bytes().replace(b'TEST_ONLY_ACTION',b'TEST_ONLY_ACTIOX'))
    if fault=='version_roster':m=fixture[3]({**m['body'],'version_ids':['v2']})
    with pytest.raises((PermissionError,ValueError,AssertionError)):accept(p,m,reg,i,test_only=True)

def split_row(fixture,status='a'):
    r=copy.deepcopy(fixture[0]);r.update(kind='split',field='splitFactor',unit='share_ratio');r['values']={'ticker':'IEF','permaTicker':'TEST_ONLY_IEF','exDate':'2023-02-01','splitFrom':1.,'splitTo':2.,'splitFactor':2.,'splitStatus':status,'currency':'USD'};return r
@pytest.mark.parametrize('fault',['ratio','zero','status','nan'])
def test_split_fail_closed(fixture,fault):
    r=split_row(fixture)
    if fault=='ratio':r['values']['splitFactor']=.5
    if fault=='zero':r['values']['splitFrom']=0
    if fault=='status':r['values']['splitStatus']='unknown'
    if fault=='nan':r['values']['splitTo']=float('nan')
    with pytest.raises((ValueError,PermissionError)):run(fixture,[r])
@pytest.mark.parametrize('status',['a','c'])
def test_split_active_cancelled_retained_without_execution(fixture,status):
    r=split_row(fixture,status);out=run(fixture,[r]);assert not out['results'][0]['P1_qualified'];v=reconcile(out['records'],[],[],'2023-12-31');assert not v['rows'][0]['share_change_allowed'];assert ('CANCELLED' in v['rows'][0]['state'])==(status=='c')
@pytest.mark.parametrize('missing',['paymentDate','recordDate','declarationDate'])
def test_null_official_date_is_partial_not_invented(fixture,missing):
    r=copy.deepcopy(fixture[0]);r['values'][missing]=None;out=run(fixture,[r]);assert missing in out['results'][0]['missing'] and out['qualified_Tier_A']==0
@pytest.mark.parametrize('complete',[True,False])
def test_no_action_requires_explicit_complete_proof(fixture,complete):
    r=copy.deepcopy(fixture[0]);r.update(kind='no_action',field='no_action',unit='coverage');r['values']={'start':'2023-01-01','end':'2023-12-31','covered_types':['dividend','split'],'complete':complete,'explicit_no_action':complete,'coverage_record_sha256':'a'*64};out=run(fixture,[r]);assert not out['results'][0]['P1_qualified'] and out['qualified_Tier_A']==0
@pytest.mark.parametrize('known,pub',[ (None,None),('2023-02-02T12:00:00Z','2023-02-02T12:00:00Z')])
def test_cash_effective_date_is_not_asof(fixture,known,pub):
    r=copy.deepcopy(fixture[0]);r.update(kind='cash',field='percentRate',unit='percent_annualized',known_at=known);r['values']={'effectiveDate':'2023-02-01','percentRate':4.5,'instrument':'TEST_ONLY_CASH','day_count':'ACT/360','decision_at':'2023-02-02T00:00:00Z','publication_at':pub,'cash_execution_attested':False};out=run(fixture,[r]);assert 'cash_authenticated_known_by_decision' in out['results'][0]['missing']
@pytest.mark.parametrize('fault',['modified2024','adjusted','nullpublication'])
def test_eod_correction_or_basis_not_execution_evidence(fixture,fault):
    r=copy.deepcopy(fixture[0]);r.update(kind='raw_open',field='open',unit='USD');r['values']={'session_date':'2023-02-01','open':100.,'price_basis':'raw_unadjusted','modified_at':'2023-02-02T20:00:00Z','publication_at':None,'permission_attested':False}
    if fault=='modified2024':r['values']['modified_at']='2024-01-01T00:00:00Z'
    elif fault=='adjusted':r['values']['price_basis']='adjusted'
    if fault=='nullpublication':assert not run(fixture,[r])['results'][0]['P1_qualified']
    else:
        with pytest.raises((ValueError,PermissionError)):run(fixture,[r])

def test_outside_inbox_rejected_before_read(fixture):
    p,m,reg,i=fixture[4]()
    with patch.object(Path,'read_bytes',side_effect=AssertionError('Outside inbox read')):
        with pytest.raises(PermissionError):accept(i.parent/'original.json',m,reg,i,test_only=True)

def test_import_idempotency_conflict_and_append_only(fixture,tmp_path):
    out=run(fixture);db=tmp_path/'new.sqlite';assert register_import(db,out)['state'].startswith('RECORDED');assert register_import(db,out)['state']=='IDEMPOTENT_REUSE'
    wrong={**out,'payload_sha256':'b'*64}
    with pytest.raises(ValueError):register_import(db,wrong)
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM imports').fetchone()[0]==1
        with pytest.raises(sqlite3.DatabaseError):c.execute('DELETE FROM imports')

def test_reconcile_match_conflict_cancel_is_not_new_actual_evidence(fixture):
    r=copy.deepcopy(fixture[0]);legacy=[{'asset':'IEF','ex_date':'2023-02-01','divCash':.212178}];issuer=[{'asset':'IEF','ex_date':'2023-02-01','amount':.212178,'pay_date':'2023-02-08'}];v=reconcile([r],legacy,issuer,'2023-12-31');assert v['matched_legacy_rows']==1 and v['issuer_payment_conflicts']==1 and v['new_real_provider_matches']==0
    r['values']['distributionFrequency']='c';v=reconcile([r],legacy,issuer,'2023-12-31');assert v['matched_legacy_rows']==0 and v['rows'][0]['payment_status']=='CANCELLED'

def test_real_empty_registry_rejects_even_fixture_signature(fixture):
    p,m,reg,i=fixture[4]();real={'schema':'v37-trust-1','environment':'REAL','independently_enrolled':False,'custodians':{},'publishers':{},'reviewers':{}}
    with patch.object(Path,'read_bytes',side_effect=AssertionError('Not authorized')):
        with pytest.raises(PermissionError):accept(p,m,real,i)

@pytest.mark.parametrize('unit',['percent','USD','contracts'])
def test_wrong_distribution_unit_rejected(fixture,unit):
    r=copy.deepcopy(fixture[0]);r['unit']=unit
    with pytest.raises(ValueError,match='unit'):run(fixture,[r])

def test_duplicate_semantic_action_different_event_rejected(fixture):
    r=copy.deepcopy(fixture[0]);other=copy.deepcopy(r);other['event_id']='duplicate_alias'
    with pytest.raises(ValueError,match='Ambiguous'):run(fixture,[r,other])

def test_same_event_explicit_revision_versions_retained(fixture):
    r=copy.deepcopy(fixture[0]);other=copy.deepcopy(r);other['version_id']='v2';other['values']['paymentDate']='2023-02-08';other['correction_chain']=[{'version_id':'v1','payload_sha256':'a'*64},{'version_id':'v2','payload_sha256':'b'*64}]
    out=run(fixture,[r,other]);assert out['rows']==2 and out['qualified_Tier_A']==0

def test_reviewer_cannot_share_custodian_key(fixture):
    from qualification_v35.core import EVIDENCE_FIELDS
    r=copy.deepcopy(fixture[0]);r.update(kind='source_evidence',field='documentary_release',unit='index')
    e={k:None for k in EVIDENCE_FIELDS};e.update(schema_version='v35-evidence-1',claim_id='TEST_ONLY_SOURCE',source=r['provider'],publisher=r['publisher'],url='https://fixture.invalid/no-network',entity=r['entity'],field=r['field'],version_id=r['version_id'],payload_sha256=sha(b'TEST_ONLY'),parser_sha256='a'*64,valid_at=r['valid_at'],known_at=None,first_public_at=None,retrieved_at='2023-12-31T23:59:59Z',clock_basis='retrieval',version_kind='original',correction_chain=[],correction_history_complete=False,scope=['TEST_ONLY'],scope_complete=False,test_not_evidence=True)
    r['values']={'claim':e,'raw_payload_b64':base64.b64encode(b'TEST_ONLY').decode(),'clock_payload_b64':None};p,m,reg,i=fixture[4]([r]);reg=copy.deepcopy(reg);key=reg['custodians']['TEST_ONLY_CUSTODIAN']['public_key_hex'];reg['reviewers']['same_key_different_name']={'role':'independent_reviewer','public_key_hex':key,'enrollment_evidence_sha256':'b'*64,'providers':[r['provider']],'publishers':[r['publisher']],'test_not_evidence':True}
    with pytest.raises(PermissionError,match='Reviewer'):accept(p,m,reg,i,test_only=True)

def test_false_no_action_conflicts_with_active_dividend(fixture):
    r=copy.deepcopy(fixture[0]);n=copy.deepcopy(r);n.update(kind='no_action',field='no_action',unit='coverage',event_id='no-action');n['values']={'start':'2023-01-01','end':'2023-12-31','covered_types':['dividend','split'],'complete':True,'explicit_no_action':True,'coverage_record_sha256':'a'*64}
    with pytest.raises(ValueError,match='conflicts'):run(fixture,[r,n])

def test_revision_current_version_must_match(fixture):
    r=copy.deepcopy(fixture[0]);r['correction_chain']=[{'version_id':'wrong','payload_sha256':'a'*64}]
    with pytest.raises(ValueError,match='current version'):run(fixture,[r])

def test_entity_roster_before_kind_processing(fixture):
    r=copy.deepcopy(fixture[0]);r['entity']='UNKNOWN'
    with pytest.raises(ValueError,match='roster'):run(fixture,[r])

def test_independent_cli_receipt_comparison_excludes_only_display_path():
    from evidence_procurement_v37.independent import O,load,receipt_matches
    out=load(O/'executed_import_demo.json')['accepted_cli'][0]
    assert receipt_matches(out['receipt'],out)
    changed=copy.deepcopy(out);changed['qualified_Tier_A']=1
    assert not receipt_matches(out['receipt'],changed)
    changed=copy.deepcopy(out);changed['unexpected']=True
    assert not receipt_matches(out['receipt'],changed)
