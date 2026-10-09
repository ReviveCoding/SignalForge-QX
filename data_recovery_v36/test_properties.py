"""V36 real software/property probes; fixtures deliberately TEST_ONLY."""
import copy,json,sqlite3,concurrent.futures
from datetime import datetime,timedelta,timezone
import pytest
from data_recovery_v36.core import *
from data_recovery_v36.shadow import *
@pytest.fixture
def plan():return {'assets':['SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'],'seeds':[11,37,71],**{k+'s':['a'*64] for k in HASH_KEYS}}
def message(n=0):
    d=datetime(2021,1,8,23,tzinfo=timezone.utc)+timedelta(weeks=n)
    return {'kind':'TEST_ONLY','retry_key':'fixture-'+str(n),'track':'Main-A','asset':'IEF','seed':11,'decision_at':d.isoformat(),'source_valid_at':(d-timedelta(days=1)).isoformat(),'source_known_at':(d-timedelta(hours=1)).isoformat(),'label_end':(d+timedelta(days=7)).isoformat(),'label_available_at':(d+timedelta(days=8)).isoformat(),'expiry_at':(d+timedelta(days=7)).isoformat(),'quantiles':[-2.,-1.,0.,1.,2.],'block':int(n>=52),**{k:'a'*64 for k in HASH_KEYS}}
@pytest.mark.parametrize('date',['2024-01-01','2025-02-01','2026-10-08'])
def test_reserved_document_date(date):
    with pytest.raises(PermissionError):historical_date(date)
@pytest.mark.parametrize('date',['2023-12-31','2010-07-20','2023-01-01'])
def test_historical_dates(date):assert historical_date(date)<BORDER
@pytest.mark.parametrize('v',[float('nan'),float('inf'),True])
def test_nonfinite_observation(v):
    with pytest.raises(ValueError):observation('EIA','US','stock','2023-08-11',v,'million_barrels','a'*64,'v1')
def test_publisher_hash_link():
    a=observation('EIA','US','stock','2023-08-11',1.,'million_barrels','a'*64,'v1');b=copy.deepcopy(a);b['unit']='thousand_barrels';assert strict_compare(a,b)['state']=='BLOCKED_IDENTITY_UNIT_MISMATCH'
def test_actual_difference_retained():
    a=observation('EIA','US','stock','2023-08-11',1.,'million_barrels','a'*64,'v1');b=copy.deepcopy(a);b['value']=1.2;assert strict_compare(a,b)['state']=='DOCUMENTARY_DISCREPANCY'
def test_external_admission_before_file_read(tmp_path):
    with pytest.raises(PermissionError):import_scoped_export(tmp_path/'NEVER_OPENED',{}, {},tmp_path)
def test_no_action_absence_not_proof():assert action_join({'asset':'IEF','ex_date':'2023-02-01','pay_date':'2023-02-07','amount':.2},[])['matches']==0
def test_ambiguous_action_join():
    x={'asset':'IEF','ex_date':'2023-02-01','pay_date':'2023-02-07','amount':.2};p={'asset':'IEF','ex_date':x['ex_date'],'divCash':.2};assert action_join(x,[p,p])['state']=='BLOCKED_AMBIGUOUS_ACTION_MATCH'
def test_pay_before_ex_rejected():
    with pytest.raises(ValueError):action_join({'asset':'IEF','ex_date':'2023-02-07','pay_date':'2023-02-01','amount':.2},[])
@pytest.mark.parametrize('split',[.25,.5,1.,2.,4.])
def test_cash_share_conservation_property(split):
    for shares in [0.,1.,17.,100.]:
        for dividend in [0.,.1,1.]:
            x=test_action_accounting(shares,100.,dividend,split,15.,5.,7);assert abs(x['before']-x['after_ex'])<1e-10 and abs(x['before']-x['after_pay'])<1e-10;assert x['after_cash_accrual']-x['after_pay']==pytest.approx(x['interest']);assert x['test_only'] and not x['real_PnL']
# Prevent imported accounting helper being collected as a test function.
test_action_accounting.__test__=False
@pytest.mark.parametrize('fault',['real','schema','hash','asset','seed','known','timezone','crossing','nan','block'])
def test_shadow_fail_no_output(tmp_path,plan,fault):
    m=message();p=tmp_path/'shadow.sqlite'
    if fault=='real':m['kind']='REAL'
    if fault=='schema':m['extra']=1
    if fault=='hash':m['model_hash']='b'*64
    if fault=='asset':m['asset']='BAD'
    if fault=='seed':m['seed']=103
    if fault=='known':m['source_known_at']='2023-01-01T00:00:00Z'
    if fault=='timezone':m['decision_at']='2021-01-08T23:00:00'
    if fault=='crossing':m['quantiles'][2]=10
    if fault=='nan':m['quantiles'][2]=float('nan')
    if fault=='block':m['block']=True
    with pytest.raises((ValueError,PermissionError)):ShadowRecorder(p,plan).append(m)
    assert not p.exists()
def test_append_idempotent(tmp_path,plan):
    p=tmp_path/'s.sqlite';r=ShadowRecorder(p,plan);a=r.append(message());b=r.append(message());assert a['id']==b['id']==1 and b['reused'];assert replay(p)['records']==1
@pytest.mark.parametrize('fault',['retry','semantic','clock'])
def test_conflicts_do_not_append(tmp_path,plan,fault):
    p=tmp_path/'s.sqlite';r=ShadowRecorder(p,plan,clock_fn=lambda:'2026-10-08T20:00:00Z');r.append(message());m=message(1)
    if fault=='retry':m['retry_key']='fixture-0'
    if fault=='semantic':m=message();m['retry_key']='different-key';m['decision_at']='2021-01-08T23:00:00Z'
    if fault=='clock':r.clock_fn=lambda:'2026-10-08T19:00:00Z'
    with pytest.raises((ValueError,sqlite3.IntegrityError)):r.append(m)
    assert replay(p)['records']==1
@pytest.mark.parametrize('mutation',['update','delete','tamper'])
def test_immutability_and_replay(tmp_path,plan,mutation):
    p=tmp_path/'s.sqlite';ShadowRecorder(p,plan).append(message())
    with sqlite3.connect(p) as db:
        if mutation in ['update','delete']:
            with pytest.raises(sqlite3.IntegrityError):db.execute('UPDATE records SET body=\'{}\'' if mutation=='update' else 'DELETE FROM records')
        else:
            db.execute('DROP TRIGGER deny_update');db.execute('UPDATE records SET body=\'{}\'')
    if mutation=='tamper':
        with pytest.raises(ValueError):replay(p)
    else:assert replay(p)['records']==1

def test_atomic_interruption_recovery(tmp_path,plan):
    p=tmp_path/'s.sqlite'
    def fail():raise RuntimeError('TEST_ONLY injected interruption')
    with pytest.raises(RuntimeError):ShadowRecorder(p,plan,clock_fn=fail).append(message())
    assert replay(p)['records']==0;assert ShadowRecorder(p,plan).append(message())['id']==1

def test_concurrent_atomic_append(tmp_path,plan):
    p=tmp_path/'s.sqlite';ShadowRecorder(p,plan).append(message())
    def append(n):return ShadowRecorder(p,plan).append(message(n))['id']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:ids=list(executor.map(append,range(1,9)))
    assert set(ids)==set(range(2,10)) and replay(p)['records']==9

def test_two_52_date_fixture_completeness(tmp_path,plan):
    p=tmp_path/'s.sqlite';r=ShadowRecorder(p,plan)
    for n in range(104):r.append(message(n))
    assert replay(p)['records']==104
    with sqlite3.connect(p) as db:body=[json.loads(x[0]) for x in db.execute('select body from records')]
    assert [len({x['decision_at'] for x in body if x['block']==b}) for b in [0,1]]==[52,52]

def test_cash_archive_scope_and_duplicate():
    p=json.dumps({'refRates':[{'type':'SOFR','effectiveDate':'2023-01-03','percentRate':4.3}]}).encode();assert cash_rows(p,'a'*64)[0]['unit']=='percent_annualized'
    with pytest.raises(ValueError):cash_rows(json.dumps({'refRates':json.loads(p)['refRates']*2}).encode(),'a'*64)
def test_clock_claim_is_not_proof():assert not documentary_tier({'sha256':'a'*64,'family':'BLS','url':'https://www.bls.gov/x'},'embargoed until 8:30')['first_public']
