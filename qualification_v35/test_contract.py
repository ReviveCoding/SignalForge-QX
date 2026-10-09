"""Synthetic contract probes, never original-publication or prospective evidence."""
import base64,copy,json,subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from qualification_v35.core import *

@pytest.fixture
def signer(tmp_path):
    key=tmp_path/'fixture_key.pem';subprocess.run(['openssl','genpkey','-algorithm','ED25519','-out',str(key)],check=True,capture_output=True)
    pub=subprocess.check_output(['openssl','pkey','-in',str(key),'-pubout','-outform','DER'])[-32:].hex()
    def sign(body):
        (tmp_path/'body').write_bytes(canonical(body));subprocess.run(['openssl','pkeyutl','-sign','-inkey',str(key),'-rawin','-in',str(tmp_path/'body'),'-out',str(tmp_path/'sig')],check=True,capture_output=True)
        return {'body':body,'key_id':'TEST_NOT_EVIDENCE','signature_b64':base64.b64encode((tmp_path/'sig').read_bytes()).decode()}
    return sign,{'TEST_NOT_EVIDENCE':pub}

def claim(signer):
    sign,keys=signer;payload=b'TEST_NOT_EVIDENCE synthetic release';stamp=b'TEST_NOT_EVIDENCE dissemination'
    e={k:None for k in EVIDENCE_FIELDS};e.update(schema_version='v35-evidence-1',claim_id='test1',source='fixture',publisher='fixture_publisher',url='https://example.invalid/fixture',entity='IEF',field='synthetic_only',version_id='1',payload_sha256=sha(payload),parser_sha256='a'*64,valid_at='2023-01-01T00:00:00Z',known_at='2023-01-02T10:00:00Z',first_public_at='2023-01-02T10:00:00Z',retrieved_at='2023-01-03T00:00:00Z',clock_basis='authenticated_first_public',version_kind='original',correction_chain=[{'version_id':'1','payload_sha256':sha(payload)}],correction_history_complete=True,scope=['IEF:2023:test1'],scope_complete=True,test_not_evidence=True)
    e['dissemination_record']=sign({'domain':'v35-first-dissemination-1','payload_sha256':sha(payload),'dissemination_payload_sha256':sha(stamp),'version_id':'1','first_public_at':e['first_public_at'],'publisher':e['publisher'],'entity':e['entity'],'field':e['field']})
    e['reviewer_record']=sign({'domain':'v35-independent-evidence-review-1','evidence_hash':sha(canonical({k:v for k,v in e.items() if k!='reviewer_record'}))})
    return e,payload,stamp,keys

def test_positive_crypto_fixture_never_evidence(signer):
    e,p,s,k=claim(signer);assert qualify(e,p,s,k,k)['state']=='QUALIFIED_TEST_FIXTURE'
@pytest.mark.parametrize('basis',sorted(CLOCK_BASES-{'authenticated_first_public'}))
def test_spoofed_clocks(signer,basis):
    e,p,s,k=claim(signer);e['clock_basis']=basis;assert qualify(e,p,s,k,k)['state']=='VERIFIED_PAYLOAD_BUT_NO_FIRST_PUBLIC'
@pytest.mark.parametrize('fault,state',[('wronghash','BLOCKED_CONFLICT'),('missingraw','BLOCKED_MISSING_ORIGINAL'),('timezone','BLOCKED_CLOCK'),('revisions','BLOCKED_REVISIONS'),('schema','BLOCKED_SCHEMA'),('trust','BLOCKED_MISSING_ORIGINAL'),('version','BLOCKED_CONFLICT'),('signature','BLOCKED_MISSING_ORIGINAL'),('partial','PARTIAL_DOCUMENTARY')])
def test_evidence_faults(signer,fault,state):
    e,p,s,k=claim(signer)
    if fault=='wronghash':p=b'changed'
    if fault=='missingraw':p=None
    if fault=='timezone':e['known_at']='2023-01-02T10:00:00'
    if fault=='revisions':e['correction_history_complete']=False
    if fault=='schema':e['unknown']=True
    if fault=='trust':k={}
    if fault=='version':e['dissemination_record']['body']['version_id']='wrong'
    if fault=='signature':e['dissemination_record']['signature_b64']=base64.b64encode(b'0'*64).decode()
    if fault=='partial':
        e['scope_complete']=False;e['reviewer_record']=signer[0]({'domain':'v35-independent-evidence-review-1','evidence_hash':sha(canonical({a:v for a,v in e.items() if a!='reviewer_record'}))})
    assert qualify(e,p,s,k,k)['state']==state

def test_duplicate_claim(signer):
    e,p,s,k=claim(signer)
    with pytest.raises(ValueError):qualify_many([e,e],{sha(p):p},{'test1':s},k,k)
def test_signed_body_tampering(signer):
    record=signer[0]({'domain':'fixture','content':'authentic_fixture'});record['body']['content']='forged';assert not signed_record(record,signer[1],'fixture')
def test_missing_p1_is_not_no_action():
    s={'asset':'IEF','session_date':'2023-01-03','decision_at':'2022-12-30T23:00:00Z','raw_open':99.,'proofs':{},'action_status':'unknown','test_not_evidence':False};v=p1_session(s);assert not v['P1_eligible'] and 'explicit_action_status' in v['missing']
@pytest.mark.parametrize('price',[0,-1,float('nan'),float('inf'),True])
def test_invalid_raw_open(price):
    with pytest.raises(ValueError):p1_session({'asset':'IEF','session_date':'2023-01-03','decision_at':'2022-12-30T23:00:00Z','raw_open':price,'proofs':{},'action_status':'unknown','test_not_evidence':True})
def test_readiness_no_schedule_without_real_freeze():
    r=readiness({});assert r['state']=='BLOCKED_EVIDENCE' and r['schedule'] is None and not r['can_issue_forecast']

def future_fixture(signer):
    p={'assets':['IEF','TLT'],'seeds':[11,37,71],'model_hashes':['a'*64],'source_hashes':['b'*64],'normalizer_hash':'c'*64,'target_hash':'d'*64}
    f=signer[0]({'domain':'v35-operational-freeze-1','plan_hash':sha(canonical(p)),'frozen_at':'2030-01-01T00:00:00Z','sources_qualified':True,'models_qualified':True})
    rows=[]
    for j in range(104):
        d=datetime(2030,1,4,23,tzinfo=timezone.utc)+timedelta(weeks=j)
        for a in p['assets']:
            for seed in p['seeds']:
                r={'decision_at':d.isoformat(),'issued_at':(d+timedelta(seconds=5)).isoformat(),'input_available_at':(d-timedelta(hours=1)).isoformat(),'label_end':(d+timedelta(days=7)).isoformat(),'label_available_at':(d+timedelta(days=8)).isoformat(),'expiry_at':(d+timedelta(days=7)).isoformat(),'asset':a,'seed':seed,'model_hash':'a'*64,'source_hash':'b'*64,'normalizer_hash':'c'*64,'target_hash':'d'*64,'block':int(j>=52),'quantiles':[-2.,-1.,0.,1.,2.],'test_not_evidence':True}
                r['preoutcome_receipt_hash']=sha(canonical({k:v for k,v in r.items() if k not in ['label_end','label_available_at','test_not_evidence']}));rows.append(r)
    return p,f,rows

def test_future_104_independent_dates(signer):
    p,f,r=future_fixture(signer);v=validate_forward(r,f,p,'2033-01-01T00:00:00Z',synthetic=True,reviewer_keys=signer[1]);assert v['state']=='TEST_FIXTURE_COMPLETE' and v['qualified_dates']==104 and not v['actual_future_evidence']
@pytest.mark.parametrize('fault',['missing_freeze','real_access','immature','backdated','duplicate','missing_asset','source','tamper','overlap','timezone'])
def test_forward_fail_closed(signer,fault):
    p,f,r=future_fixture(signer);synthetic=True
    if fault=='missing_freeze':f=None
    if fault=='real_access':synthetic=False
    if fault=='immature':r[0]['label_available_at']='2035-01-01T00:00:00Z'
    if fault=='backdated':r[0]['issued_at']='2029-01-01T00:00:00Z'
    if fault=='duplicate':r.append(copy.deepcopy(r[0]))
    if fault=='missing_asset':r.pop(0)
    if fault=='source':r[0]['source_hash']='e'*64
    if fault=='tamper':r[0]['quantiles'][2]=99
    if fault=='overlap':r[0]['block']=1
    if fault=='timezone':r[0]['decision_at']='2030-01-04T23:00:00'
    assert validate_forward(r,f,p,'2033-01-01T00:00:00Z',synthetic=synthetic,reviewer_keys=signer[1])['state']=='BLOCKED'

def test_p1_signed_complete_fixture_and_unknown_authority(signer):
    sign,keys=signer;s={'asset':'IEF','session_date':'2023-01-03','decision_at':'2022-12-30T23:00:00Z','raw_open':99.,'proofs':{},'action_status':'no_action_attested','test_not_evidence':True}
    for k in P1_REQUIREMENTS:
        body={'domain':'v35-P1-proof-1','asset':'IEF','session_date':'2023-01-03','requirement':k,'receipt_sha256':'a'*64,'known_at':'2022-12-30T22:00:00Z'}
        s['proofs'][k]={'qualified':True,'receipt_sha256':'a'*64,'attestation':sign(body)}
    assert p1_session(s,keys)['state']=='QUALIFIED_TEST_FIXTURE';assert not p1_session(s,keys)['P1_eligible'];assert len(p1_session(s)['missing'])==10
    s['proofs']['asof_cash']['attestation']=sign({'domain':'v35-P1-proof-1','asset':'IEF','session_date':'2023-01-03','requirement':'asof_cash','receipt_sha256':'a'*64,'known_at':'2023-01-04T00:00:00Z'})
    assert 'asof_cash' in p1_session(s,keys)['missing']

def test_revision_requires_parent(signer):
    e,p,s,k=claim(signer);e['version_kind']='revision';assert qualify(e,p,s,k,k)['state']=='BLOCKED_REVISIONS'
def test_no_claim_can_self_enroll_key(signer):
    e,p,s,k=claim(signer);e['test_not_evidence']=False;assert qualify(e,p,s)['state']=='BLOCKED_MISSING_ORIGINAL'
def test_synthetic_dividend_receivable_conservation():
    # TEST_NOT_EVIDENCE: held-before-ex shares, receivable on ex, paid later.
    shares=10.;raw_open=100.;cash=0.;receivable=0.;equity0=shares*raw_open
    raw_open-=1.;receivable+=shares*1.;assert shares*raw_open+receivable+cash==equity0
    cash+=receivable;receivable=0.;assert shares*raw_open+cash==equity0
    # Split changes share units, not account value.
    shares*=2.;raw_open/=2.;assert shares*raw_open+cash==equity0

def test_fixture_calendar_and_no_payment_before_ex():
    ex=clock('2023-01-03T14:30:00Z');pay=clock('2023-01-09T14:30:00Z');assert pay>ex and ex.weekday()<5 and pay.weekday()<5
    assert not clock('2023-01-02T14:30:00Z')>=ex
