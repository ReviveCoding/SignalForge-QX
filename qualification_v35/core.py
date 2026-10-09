"""V35 conservative evidence/forward validation. No acquisition, training or issuance."""
import base64,hashlib,json,math,re,subprocess,tempfile
from datetime import datetime,timezone
from pathlib import Path

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def clock(s):
    if not isinstance(s,str):raise ValueError('Explicit aware clock required')
    t=datetime.fromisoformat(s.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('Timezone ambiguity')
    return t.astimezone(timezone.utc)
def valid_sha(s):return isinstance(s,str) and re.fullmatch('[0-9a-f]{64}',s) is not None

def signed_record(record,trusted_keys,domain):
    """Externally enrolled public keys only. No self-declared trust or generated authority."""
    try:
        if set(record)!={'body','key_id','signature_b64'}:return False
        body=record['body']; key=trusted_keys[record['key_id']]
        if body.get('domain')!=domain or len(bytes.fromhex(key))!=32:return False
        with tempfile.TemporaryDirectory(prefix='v35_signature_') as d:
            d=Path(d);(d/'pub.der').write_bytes(bytes.fromhex('302a300506032b6570032100')+bytes.fromhex(key));(d/'body').write_bytes(canonical(body));(d/'sig').write_bytes(base64.b64decode(record['signature_b64'],validate=True))
            p=subprocess.run(['openssl','pkeyutl','-verify','-pubin','-keyform','DER','-inkey',str(d/'pub.der'),'-rawin','-in',str(d/'body'),'-sigfile',str(d/'sig')],capture_output=True,timeout=10)
            return p.returncode==0
    except (KeyError,ValueError,TypeError,subprocess.SubprocessError):return False

EVIDENCE_FIELDS={'schema_version','claim_id','source','publisher','url','entity','field','version_id','payload_sha256','parser_sha256','valid_at','known_at','first_public_at','retrieved_at','clock_basis','dissemination_record','version_kind','correction_chain','correction_history_complete','scope','scope_complete','reviewer_record','test_not_evidence'}
CLOCK_BASES={'authenticated_first_public','retrieval','pdf_date','scheduled_release','last_modified','edgar_acceptance','database_vintage','reconstructed_eod','inferred'}
def qualify(e,payload,clock_payload,keys=None,reviewers=None):
    keys=keys or {};reviewers=reviewers or {}
    def result(state,reason):return {'claim_id':e.get('claim_id'),'state':state,'reason':reason,'scientific_evidence':not e.get('test_not_evidence',False),'qualified_for_final':False}
    if set(e)!=EVIDENCE_FIELDS or e.get('schema_version')!='v35-evidence-1':return result('BLOCKED_SCHEMA','Strict field set/version required')
    if not all(e.get(k) for k in ['claim_id','source','publisher','url','entity','field','version_id','scope']):return result('BLOCKED_SCHEMA','Identity/scope missing')
    if not all(valid_sha(e[k]) for k in ['payload_sha256','parser_sha256']):return result('BLOCKED_SCHEMA','Exact payload/parser hashes required')
    if payload is None:return result('BLOCKED_MISSING_ORIGINAL','Original payload absent')
    if sha(payload)!=e['payload_sha256']:return result('BLOCKED_CONFLICT','Raw bytes hash mismatch')
    try:
        valid=clock(e['valid_at']);known=clock(e['known_at']);retrieved=clock(e['retrieved_at'])
        if known>retrieved:raise ValueError('Known after retrieval')
    except (ValueError,TypeError):return result('BLOCKED_CLOCK','Invalid or ambiguous clock')
    if e['clock_basis'] not in CLOCK_BASES:return result('BLOCKED_CLOCK','Unregistered clock semantics')
    if e['clock_basis']!='authenticated_first_public':return result('VERIFIED_PAYLOAD_BUT_NO_FIRST_PUBLIC','Content verified; clock is '+e['clock_basis'])
    try:
        first=clock(e['first_public_at']);assert first<=known and first<=retrieved
        record=e['dissemination_record'];assert clock_payload is not None
        assert record['body']['dissemination_payload_sha256']==sha(clock_payload)
        assert record['body']['payload_sha256']==e['payload_sha256'] and record['body']['version_id']==e['version_id']
        assert record['body']['first_public_at']==e['first_public_at'] and record['body']['publisher']==e['publisher']
        assert record['body']['entity']==e['entity'] and record['body']['field']==e['field']
    except (KeyError,AssertionError,ValueError,TypeError):return result('BLOCKED_CONFLICT','Clock record/version linkage missing or inconsistent')
    if not signed_record(record,keys,'v35-first-dissemination-1'):return result('BLOCKED_MISSING_ORIGINAL','Trusted authenticated publisher dissemination record absent')
    chain=e['correction_chain']
    if e['version_kind'] not in ['original','revision'] or e['correction_history_complete'] is not True or not chain:return result('BLOCKED_REVISIONS','Complete correction lineage absent')
    if len({v.get('version_id') for v in chain})!=len(chain) or any(not valid_sha(v.get('payload_sha256')) for v in chain):return result('BLOCKED_REVISIONS','Duplicate or malformed revision chain')
    if chain[-1]!={'version_id':e['version_id'],'payload_sha256':e['payload_sha256']} or (e['version_kind']=='revision' and len(chain)<2):return result('BLOCKED_REVISIONS','Current version or original parent missing')
    review=e['reviewer_record']
    if not signed_record(review,reviewers,'v35-independent-evidence-review-1') or review['body'].get('evidence_hash')!=sha(canonical({k:v for k,v in e.items() if k!='reviewer_record'})):return result('PARTIAL_DOCUMENTARY','Independent reviewed identity absent')
    if e['scope_complete'] is not True:return result('PARTIAL_DOCUMENTARY','Coverage scope incomplete')
    return result('QUALIFIED_TEST_FIXTURE' if e['test_not_evidence'] else 'QUALIFIED_TIER_A','Authenticated version-specific evidence')

def qualify_many(claims,payloads,clock_payloads,keys=None,reviewers=None):
    ids=[e.get('claim_id') for e in claims]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate claim identity')
    return [qualify(e,payloads.get(e['payload_sha256']),clock_payloads.get(e['claim_id']),keys,reviewers) for e in claims]

P1_REQUIREMENTS=['raw_execution_open','open_publication','action_or_explicit_no_action','complete_actions_and_splits','actual_ex_pay_dates','action_known_by_decision','asof_cash','permission','cost_fill_convention','accounting_oracles']
def p1_session(s,trusted_keys=None):
    required={'asset','session_date','decision_at','raw_open','proofs','action_status','test_not_evidence'}
    if set(s)!=required:raise ValueError('Strict P1 session schema')
    clock(s['decision_at']);datetime.strptime(s['session_date'],'%Y-%m-%d')
    if isinstance(s['raw_open'],bool) or not isinstance(s['raw_open'],(float,int)) or not math.isfinite(s['raw_open']) or s['raw_open']<=0:raise ValueError('Raw unadjusted execution open required')
    missing=[]
    for k in P1_REQUIREMENTS:
        p=s['proofs'].get(k,{})
        record=p.get('attestation',{}); body=record.get('body',{})
        ok=p.get('qualified') is True and valid_sha(p.get('receipt_sha256')) and signed_record(record,trusted_keys or {},'v35-P1-proof-1') and body.get('asset')==s['asset'] and body.get('session_date')==s['session_date'] and body.get('requirement')==k and body.get('receipt_sha256')==p.get('receipt_sha256')
        if ok and k in ['action_known_by_decision','asof_cash']:
            try:ok=clock(body['known_at'])<=clock(s['decision_at'])
            except (KeyError,ValueError,TypeError):ok=False
        if not ok:missing.append(k)
    if s['action_status'] not in ['event_attested','no_action_attested']:missing.append('explicit_action_status')
    state='QUALIFIED_TEST_FIXTURE' if not missing and s['test_not_evidence'] else 'QUALIFIED_P1' if not missing else 'VERIFIED_PRICE_PARTIAL'
    return {'state':state,'missing':missing,'P1_eligible':not missing and not s['test_not_evidence']}

def readiness(flags):
    required=['candidate_hashes_verified','benchmark_hashes_verified','normalizers_verified','all_source_versions_tier_a','all_P1_sessions_qualified','independent_reviewer','thresholds_mature_train_only','operational_freeze_verified','scientific_freeze_verified','user_candidate_selection']
    missing=[k for k in required if flags.get(k) is not True]
    return {'state':'READY_FOR_INDEPENDENT_HANDOFF' if not missing else 'BLOCKED_EVIDENCE','missing':missing,'can_issue_forecast':False,'schedule':None,'new_forecasts':0}

def validate_forward(rows,freeze,plan,now_at,*,synthetic=False,reviewer_keys=None):
    """Pure validator; NEVER reads labels/files, issues forecasts or signs a freeze."""
    fail=lambda x:{'state':'BLOCKED','reason':x,'actual_future_evidence':False,'qualified_dates':0}
    if not freeze:return fail('Missing independently verified operational freeze')
    if not signed_record(freeze,reviewer_keys or {},'v35-operational-freeze-1'):return fail('Untrusted freeze signature')
    b=freeze['body']
    if b.get('plan_hash')!=sha(canonical(plan)) or not b.get('sources_qualified') or not b.get('models_qualified'):return fail('Freeze binding or scientific gates')
    if not synthetic:return fail('Real future access/forecast execution forbidden in v35 task')
    if not all(r.get('test_not_evidence') is True for r in rows):return fail('Fixture/evidence separation')
    try:
        ft=clock(b['frozen_at']);nt=clock(now_at);seen=set();dates=[set(),set()];last=None
        for r in rows:
            qs=r['quantiles']
            if len(qs)!=5 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in qs) or qs!=sorted(qs):return fail('Finite ordered quantile contract')
            identity=(r['decision_at'],r['asset'],r['seed'])
            if identity in seen:return fail('Duplicate immutable forecast identity')
            seen.add(identity);d=clock(r['decision_at']);issued=clock(r['issued_at']);available=clock(r['input_available_at']);end=clock(r['label_end']);mature=clock(r['label_available_at']);expiry=clock(r['expiry_at'])
            if not ft<d<=issued<end<=mature<=nt or not available<=d or not d<expiry<=end:return fail('Future clock/maturity order')
            if r['model_hash'] not in plan['model_hashes'] or r['source_hash'] not in plan['source_hashes'] or r['asset'] not in plan['assets'] or r['seed'] not in plan['seeds']:return fail('Exact source/model/cohort binding')
            if r['normalizer_hash']!=plan['normalizer_hash'] or r['target_hash']!=plan['target_hash']:return fail('Common target/normalizer binding')
            if r['preoutcome_receipt_hash']!=sha(canonical({k:v for k,v in r.items() if k not in ['preoutcome_receipt_hash','label_end','label_available_at','test_not_evidence']})):return fail('Prediction/pre-outcome integrity')
            if r['block'] not in [0,1]:return fail('Exactly two blocks')
            dates[r['block']].add(d)
        if dates[0]&dates[1] or (dates[0] and dates[1] and max(dates[0])>=min(dates[1])):return fail('Non-overlapping chronological blocks')
        for block,ds in enumerate(dates):
            for d in ds:
                actual={(r['asset'],r['seed']) for r in rows if r['block']==block and clock(r['decision_at'])==d}
                if actual!={(a,s) for a in plan['assets'] for s in plan['seeds']}:return fail('Incomplete date asset seed grid')
        counts=[len(x) for x in dates]
        return {'state':'TEST_FIXTURE_COMPLETE' if counts==[52,52] else 'TEST_FIXTURE_INCOMPLETE','date_counts':counts,'qualified_dates':sum(counts),'actual_future_evidence':False,'independent_date_not_seed_unit':True}
    except (KeyError,ValueError,TypeError):return fail('Malformed future contract')
