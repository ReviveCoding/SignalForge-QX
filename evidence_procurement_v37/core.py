"""Authorized PRE_2024_ONLY import. No HTTP, credentials, old exports or scientific freeze."""
import json,math,sqlite3,base64
from pathlib import Path
from datetime import datetime,timezone
from qualification_v35.core import canonical,sha,clock,valid_sha,signed_record,qualify,EVIDENCE_FIELDS
BORDER=clock('2024-01-01T00:00:00Z')
ASSETS={'SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'}
BODY_FIELDS={'domain','dataset_id','provider','publisher','scope','snapshot_asof','range_start','range_end','min_valid_at','max_valid_at','min_known_at','max_known_at','assets','series','row_count','payload_bytes','payload_sha256','version_ids','excluded_rows_count','independent_pre2024_export','no_opaque_attachments','license','test_not_evidence'}
ROW_FIELDS={'event_id','version_id','provider','publisher','kind','entity','field','unit','valid_at','known_at','first_public_at','clock_basis','correction_chain','values','test_not_evidence'}
LICENSE_FIELDS={'research_use_permitted','local_analysis_permitted','entitlement_confirmed','restrictions_resolved','range_start','range_end','assets','series','license_record_id','public_redistribution_permitted'}
KINDS={'dividend','split','no_action','cash','raw_open','source_evidence'}

def before(s):
    t=clock(s)
    if t>=BORDER:raise PermissionError('PRE2024_SCOPE_VIOLATION')
    return t

def day(s):
    if not isinstance(s,str):raise ValueError('Explicit calendar field')
    if len(s)==10:d=datetime.strptime(s,'%Y-%m-%d').date()
    else:d=before(s).date()
    if d.isoformat()>='2024-01-01':raise PermissionError('PRE2024_SCOPE_VIOLATION')
    return d.isoformat()

def finite(v,*,positive=False):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or (positive and v<=0):raise ValueError('Finite numeric/basis required')
    return float(v)

def admission(manifest,registry,*,test_only=False):
    # This function never opens, stats, hashes, filters or parses any dataset path.
    if registry.get('schema')!='v37-trust-1':raise PermissionError('TRUST_REGISTRY_SCHEMA')
    env='TEST_ONLY' if test_only else 'REAL'
    if registry.get('environment')!=env or registry.get('independently_enrolled') is not True:raise PermissionError('TRUST_NOT_INDEPENDENTLY_ENROLLED')
    custodians=registry.get('custodians',{});key=manifest.get('key_id');custodian=custodians.get(key,{})
    if custodian.get('role')!='authorized_export_custodian' or custodian.get('test_not_evidence') is not test_only or not valid_sha(custodian.get('enrollment_evidence_sha256')):raise PermissionError('UNKNOWN_OR_UNSUPPORTED_CUSTODIAN')
    body=manifest.get('body',{})
    if set(body)!=BODY_FIELDS or body.get('domain')!='v37-authorized-pre2024-export-1':raise ValueError('STRICT_MANIFEST_SCHEMA')
    if body['test_not_evidence'] is not test_only or body['provider'] not in custodian.get('providers',[]) or body['publisher'] not in custodian.get('publishers',[]):raise PermissionError('CUSTODIAN_IDENTITY_OR_ENVIRONMENT_MISMATCH')
    if not signed_record(manifest,{key:custodian.get('public_key_hex')},'v37-authorized-pre2024-export-1'):raise PermissionError('INVALID_EXTERNAL_SIGNATURE')
    if body['scope']!='PRE_2024_ONLY' or body['independent_pre2024_export'] is not True or body['no_opaque_attachments'] is not True:raise PermissionError('CERTIFIED_SCOPE_REQUIRED')
    before(body['snapshot_asof']);start=day(body['range_start']);end=day(body['range_end'])
    if start>end:raise ValueError('Scope date ordering')
    for lo,hi in [('min_valid_at','max_valid_at'),('min_known_at','max_known_at')]:
        if before(body[lo])>before(body[hi]):raise ValueError('Scope clock ordering')
    if before(body['max_known_at'])>before(body['snapshot_asof']):raise ValueError('Later-than-snapshot source version')
    for key in ['row_count','payload_bytes','excluded_rows_count']:
        if isinstance(body[key],bool) or not isinstance(body[key],int) or body[key]<0:raise ValueError('Attested integer count')
    if body['row_count']>100000 or body['payload_bytes']>12*1024*1024:raise ValueError('Bounded inbox resource safety')
    if not body['dataset_id'] or not valid_sha(body['payload_sha256']) or not isinstance(body['version_ids'],list) or len(set(body['version_ids']))!=len(body['version_ids']):raise ValueError('Dataset/hash/version identity')
    if not set(body['assets'])<=ASSETS or not isinstance(body['series'],list) or not body['assets'] and not body['series']:raise ValueError('Explicit asset/series roster')
    license=body['license']
    if set(license)!=LICENSE_FIELDS or not license['license_record_id']:raise PermissionError('LICENSE_ATTESTATION_MISSING')
    if any(license[k] is not True for k in ['research_use_permitted','local_analysis_permitted','entitlement_confirmed','restrictions_resolved']):raise PermissionError('LICENSE_OR_ENTITLEMENT_UNRESOLVED')
    if day(license['range_start'])>start or day(license['range_end'])<end or not set(body['assets'])<=set(license['assets']) or not set(body['series'])<=set(license['series']):raise PermissionError('LICENSE_SCOPE_MISMATCH')
    return body

def proof_keys(registry,role,test_only,provider,publisher):
    result={}
    for key,value in registry.get(role,{}).items():
        if provider not in value.get('providers',[]) or publisher not in value.get('publishers',[]):continue
        if value.get('test_not_evidence') is test_only and valid_sha(value.get('enrollment_evidence_sha256')) and value.get('role')==('original_publisher' if role=='publishers' else 'independent_reviewer'):
            result[key]=value['public_key_hex']
    return result

def validate_row(r,body,registry,test_only):
    if set(r)!=ROW_FIELDS or r['kind'] not in KINDS:raise ValueError('STRICT_EVENT_SCHEMA')
    if r['test_not_evidence'] is not test_only or r['provider']!=body['provider'] or r['publisher']!=body['publisher'] or r['version_id'] not in body['version_ids']:raise PermissionError('ROW_BINDING_MISMATCH')
    if not all(r[k] for k in ['event_id','entity','field','unit']):raise ValueError('Exact event identity')
    if r['entity'] not in body['assets'] and r['entity'] not in body['series']:raise ValueError('Asset/series roster mismatch')
    valid=before(r['valid_at']);known=before(r['known_at']) if r['known_at'] is not None else None
    if not before(body['min_valid_at'])<=valid<=before(body['max_valid_at']):raise ValueError('Valid clock outside attested range')
    if known is not None and not before(body['min_known_at'])<=known<=before(body['max_known_at']):raise ValueError('Known clock outside attested range')
    first=before(r['first_public_at']) if r['first_public_at'] else None
    if first is not None and (known is None or first>known):raise ValueError('First/known clock ordering')
    if r['clock_basis'] not in ['authenticated_first_public','unverified','database_vintage','retrospective_export']:raise ValueError('Clock semantics')
    chain=r['correction_chain']
    if not isinstance(chain,list) or any(set(x)!={'version_id','payload_sha256'} or not valid_sha(x['payload_sha256']) for x in chain) or len({x['version_id'] for x in chain})!=len(chain):raise ValueError('Correction lineage identity')
    expected_units={'dividend':'USD_per_share','split':'share_ratio','no_action':'coverage','cash':'percent_annualized','raw_open':'USD'}
    if r['kind'] in expected_units and r['unit']!=expected_units[r['kind']]:raise ValueError('Exact kind/unit contract')
    if chain and chain[-1]['version_id']!=r['version_id']:raise ValueError('Correction chain current version mismatch')
    value=r['values'];missing=[];effective=None;state='PARTIAL_AUTHORIZED_EXPORT';source_qualified=False
    if r['kind'] in ['dividend','split']:
        required={'ticker','permaTicker','exDate','currency'}|({'paymentDate','recordDate','declarationDate','distribution','distributionFrequency'} if r['kind']=='dividend' else {'splitFrom','splitTo','splitFactor','splitStatus'})
        if set(value)!=required:raise ValueError('STRICT_TIINGO_FIELD_SCHEMA')
        if value['ticker']!=r['entity'] or value['ticker'] not in body['assets'] or not value['permaTicker'] or value['currency']!='USD':raise ValueError('Ticker/permaticker/currency identity')
        effective=day(value['exDate'])
        if r['kind']=='dividend':
            finite(value['distribution'])
            if value['distribution']<0:raise ValueError('Nonnegative distribution required')
            dates={k:day(value[k]) if value[k] else None for k in ['paymentDate','recordDate','declarationDate']}
            missing +=[k for k,v in dates.items() if v is None]
            if dates['declarationDate'] and dates['declarationDate']>effective:raise ValueError('Declaration after ex-date')
            if dates['paymentDate'] and dates['paymentDate']<effective:raise ValueError('Pay before ex-date')
            if dates['recordDate'] and dates['paymentDate'] and dates['recordDate']>dates['paymentDate']:raise ValueError('Record after payment')
            if value['distributionFrequency'] not in ['w','bm','m','tm','q','sa','a','ir','f','u','c']:raise ValueError('Distribution status')
            state='CANCELLED_ACTION_PRESERVED' if value['distributionFrequency']=='c' else 'PARTIAL_DIVIDEND_EXPORT'
        else:
            a,b,c=[finite(value[k],positive=True) for k in ['splitFrom','splitTo','splitFactor']]
            if not math.isclose(c,b/a,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Split factor != splitTo/splitFrom')
            if value['splitStatus'] not in ['a','c']:raise ValueError('Active/cancelled status')
            state='CANCELLED_ACTION_PRESERVED' if value['splitStatus']=='c' else 'PARTIAL_SPLIT_EXPORT'
    elif r['kind']=='no_action':
        if set(value)!={'start','end','covered_types','complete','explicit_no_action','coverage_record_sha256'}:raise ValueError('No-action schema')
        a,b=day(value['start']),day(value['end']);effective=a
        if a>b or a<body['range_start'] or b>body['range_end']:raise ValueError('No-action interval')
        if value['complete'] is not True or value['explicit_no_action'] is not True or set(value['covered_types'])!={'dividend','split'} or not valid_sha(value['coverage_record_sha256']):missing.append('complete_explicit_dividend_split_no_action_attestation')
        state='PARTIAL_NO_ACTION_CONTENT_ONLY'
    elif r['kind']=='cash':
        if set(value)!={'effectiveDate','percentRate','instrument','day_count','decision_at','publication_at','cash_execution_attested'}:raise ValueError('Cash schema')
        effective=day(value['effectiveDate']);finite(value['percentRate']);decision=before(value['decision_at']);pub=before(value['publication_at']) if value['publication_at'] else None
        if value['day_count']!='ACT/360' or not value['instrument']:raise ValueError('Cash instrument/basis')
        if pub is None or known is None or pub>decision or known>decision:missing.append('cash_authenticated_known_by_decision')
        if value['cash_execution_attested'] is not True:missing.append('actual_cash_execution_account_proof')
        state='CASH_EFFECTIVE_RATE_NOT_QUALIFIED_EXECUTION'
    elif r['kind']=='raw_open':
        if set(value)!={'session_date','open','price_basis','modified_at','publication_at','permission_attested'}:raise ValueError('Raw-open schema')
        effective=day(value['session_date']);finite(value['open'],positive=True)
        if value['price_basis']!='raw_unadjusted':raise ValueError('P0-adjusted substitution forbidden')
        modified=before(value['modified_at']);pub=before(value['publication_at']) if value['publication_at'] else None
        if modified>before(body['snapshot_asof']):raise ValueError('Stale EOD later modification')
        if pub is None or known is None:missing.append('raw_execution_open_publication_clock')
        if value['permission_attested'] is not True:missing.append('execution_price_permission')
        state='RAW_OPEN_CONTENT_NOT_EXECUTION_QUALIFIED'
    else:
        if set(value)!={'claim','raw_payload_b64','clock_payload_b64'}:raise ValueError('Source evidence schema')
        e=value['claim']
        if set(e)!=EVIDENCE_FIELDS or e['test_not_evidence'] is not test_only or e['source']!=r['provider'] or e['publisher']!=r['publisher'] or e['entity']!=r['entity'] or e['field']!=r['field'] or e['version_id']!=r['version_id']:raise ValueError('Original evidence/event binding')
        for key in ['valid_at','known_at','first_public_at']:
            if e[key]!=r[key]:raise ValueError('Evidence clock binding')
        publishers=proof_keys(registry,'publishers',test_only,r['provider'],r['publisher']);reviewers=proof_keys(registry,'reviewers',test_only,r['provider'],r['publisher'])
        custodial_keys={x.get('public_key_hex') for x in registry.get('custodians',{}).values()}
        if set(reviewers.values())&(set(publishers.values())|custodial_keys):raise PermissionError('Reviewer cannot be publisher/custodian/self')
        evidence_payload=base64.b64decode(value['raw_payload_b64'],validate=True);clock_payload=base64.b64decode(value['clock_payload_b64'],validate=True) if value['clock_payload_b64'] else None
        proof=qualify(e,evidence_payload,clock_payload,publishers,reviewers);source_qualified=proof['state']=='QUALIFIED_TIER_A' and not test_only;missing.append(proof['state']) if not source_qualified else None;state=proof['state']
    if effective is not None and not body['range_start']<=effective<=body['range_end']:raise ValueError('Effective date outside licensed/ex-date study interval')
    if r['kind']!='source_evidence':
        if r['entity'] not in body['assets'] and r['entity'] not in body['series']:raise ValueError('Asset/series roster mismatch')
        if r['clock_basis']!='authenticated_first_public' or first is None:missing.append('authenticated_original_first_public_proof')
        else:missing.append('version_bound_original_payload_and_independent_review')
        missing.append('complete_P1_session_joint_proof')
    if not chain:missing.append('complete_revision_lineage')
    return {'event_id':r['event_id'],'version_id':r['version_id'],'state':state,'missing':sorted(set(missing)),'test_not_evidence':test_only,'qualified_Tier_A':source_qualified,'P1_qualified':False,'effective_date':effective}

def accept(path,manifest,registry,inbox,*,test_only=False):
    body=admission(manifest,registry,test_only=test_only) # before path resolution/read
    inbox=Path(inbox).resolve();path=Path(path).resolve()
    if not path.is_relative_to(inbox) or path==inbox or path.is_symlink():raise PermissionError('EXTERNAL_INBOX_ONLY')
    if not test_only:
        root=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/evidence_procurement_v37/inbox').resolve()
        if inbox!=root:raise PermissionError('REAL_ISOLATED_EXT4_INBOX_REQUIRED')
    # At this point authentic independent custodian has scoped exact bytes and all temporal fields.
    if path.stat().st_size!=body['payload_bytes']:raise ValueError('Attested payload length mismatch')
    payload=path.read_bytes()
    if sha(payload)!=body['payload_sha256']:raise ValueError('Exact payload SHA mismatch')
    data=json.loads(payload)
    if not isinstance(data,dict) or set(data)!={'schema','records'} or data['schema']!='v37-authorized-events-1' or len(data['records'])!=body['row_count']:raise ValueError('Strict dataset/row-count schema')
    records=data['records'];results=[validate_row(r,body,registry,test_only) for r in records];identities=[(r['event_id'],r['version_id']) for r in records]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate event/version')
    semantic={}
    for r in records:
        if r['kind'] in ['dividend','split']:
            key=(r['provider'],r['kind'],r['entity'],day(r['values']['exDate']))
            if key in semantic and semantic[key]!=r['event_id']:raise ValueError('Ambiguous same-action identity')
            semantic[key]=r['event_id']
    for n in records:
        if n['kind']=='no_action' and n['values']['complete'] and n['values']['explicit_no_action']:
            for event in records:
                if event['entity']==n['entity'] and event['kind'] in ['dividend','split']:
                    v=event['values'];cancelled=v.get('splitStatus')=='c' or v.get('distributionFrequency')=='c'
                    if not cancelled and day(n['values']['start'])<=day(v['exDate'])<=day(n['values']['end']):raise ValueError('No-action interval conflicts with reported action')
    if set(r['version_id'] for r in records)!=set(body['version_ids']):raise ValueError('Version roster incomplete')
    return {'state':'ACCEPTED_TEST_ONLY' if test_only else 'ACCEPTED_AUTHORIZED_PARTIAL_EXPORT','dataset_id':body['dataset_id'],'provider':body['provider'],'payload_sha256':body['payload_sha256'],'manifest_sha256':sha(canonical(manifest)),'rows':len(records),'results':results,'qualified_Tier_A':sum(r['qualified_Tier_A'] for r in results),'P1_qualified_sessions':0,'real_forward_allowed':False,'test_not_evidence':test_only,'records':records}

def register_import(dbpath,result):
    path=Path(dbpath);path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS imports(identity TEXT PRIMARY KEY,payload_sha TEXT NOT NULL,manifest_sha TEXT NOT NULL,receipt TEXT NOT NULL)')
        db.execute("CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON imports BEGIN SELECT RAISE(ABORT,'Immutable'); END")
        db.execute("CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON imports BEGIN SELECT RAISE(ABORT,'Immutable'); END")
        identity=sha(canonical([result['provider'],result['dataset_id']]));existing=db.execute('SELECT payload_sha,manifest_sha FROM imports WHERE identity=?',(identity,)).fetchone()
        if existing:
            if tuple(existing)!=(result['payload_sha256'],result['manifest_sha256']):raise ValueError('Duplicate dataset identity conflict')
            return {'state':'IDEMPOTENT_REUSE','identity':identity}
        receipt={k:v for k,v in result.items() if k!='records'};db.execute('INSERT INTO imports VALUES(?,?,?,?)',(identity,result['payload_sha256'],result['manifest_sha256'],canonical(receipt).decode()));return {'state':'RECORDED_NEW_IMMUTABLE_IMPORT','identity':identity}
