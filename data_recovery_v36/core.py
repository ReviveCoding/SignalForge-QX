"""Independent pre-2024 evidence adapters. Never accesses original canonical exports."""
import json,math
from pathlib import Path
from urllib.parse import urlsplit
from qualification_v35.core import canonical,sha,clock,valid_sha,signed_record,qualify
BORDER=clock('2024-01-01T00:00:00Z')
def historical_date(s):
    t=clock(s+'T00:00:00Z') if len(s)==10 else clock(s)
    if t>=BORDER:raise PermissionError('Reserved historical scope')
    return t

def validate_scoped_export_metadata(meta,keys):
    if not signed_record(meta,keys,'v36-pre2024-export-1'):raise PermissionError('Externally certified pre-2024 export required')
    body=meta['body']
    if body.get('scope')!='PRE_2024_ONLY' or not valid_sha(body.get('payload_sha256')) or not body.get('revision_lineage_complete'):raise PermissionError('Export attestation scope/revisions')
    historical_date(body['latest_valid_at']);historical_date(body['latest_known_at'])
    return body

def import_scoped_export(path,meta,keys,allowed_inbox):
    # Signature/scope admission happens BEFORE any supplied file is opened.
    body=validate_scoped_export_metadata(meta,keys);path=Path(path).resolve();inbox=Path(allowed_inbox).resolve()
    if not path.is_relative_to(inbox):raise PermissionError('External inbox only; no original canonical export')
    payload=path.read_bytes()
    if sha(payload)!=body['payload_sha256']:raise ValueError('Custodian export hash mismatch')
    return payload

def documentary_tier(record,text=''):
    if not record.get('sha256'):return {'tier':'BLOCKED_NO_PAYLOAD','first_public':False}
    if record['family']=='ISSUER':return {'tier':'PARTIAL_CORPORATE_ACTION' if 'distribution-summary' in record['url'] else 'VERIFIED_CONTENT_ONLY','first_public':False}
    tentative='embargo' in text.lower() or 'release date' in text.lower()
    return {'tier':'POTENTIAL_FIRST_PUBLIC_BUT_UNVERIFIED' if tentative else 'VERIFIED_CONTENT_ONLY','first_public':False}

def observation(source,entity,field,reference_date,value,unit,raw_sha,version):
    historical_date(reference_date)
    if not valid_sha(raw_sha) or not version or not unit or not entity or not field:raise ValueError('Exact source identity/unit/lineage')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Finite source observation')
    return {'source':source,'entity':entity,'field':field,'reference_date':reference_date,'value':float(value),'unit':unit,'payload_sha256':raw_sha,'version_id':version,'first_public_qualified':False}

def strict_compare(a,b,tolerance=0):
    keys=['source','entity','field','reference_date','unit']
    if any(a.get(k)!=b.get(k) for k in keys):return {'state':'BLOCKED_IDENTITY_UNIT_MISMATCH','difference':None}
    diff=a['value']-b['value']
    return {'state':'MATCH' if abs(diff)<=tolerance else 'DOCUMENTARY_DISCREPANCY','difference':diff,'tolerance':tolerance,'canonical_mutated':False}

def cash_rows(payload,raw_sha):
    d=json.loads(payload);rows=d['refRates'];result=[]
    for r in rows:
        date=r['effectiveDate'];historical_date(date);v=float(r['percentRate'])
        result.append(observation('NYFED',r['type'],'percentRate',date,v,'percent_annualized',raw_sha,date))
    if len({(r['entity'],r['reference_date']) for r in result})!=len(result):raise ValueError('Duplicate effective-date rate')
    return result

def action_join(issuer,provider):
    keys=(issuer['asset'],issuer['ex_date']);historical_date(issuer['ex_date']);historical_date(issuer['pay_date'])
    if issuer['pay_date']<issuer['ex_date']:raise ValueError('Payment precedes ex-date')
    matches=[p for p in provider if (p['asset'],p['ex_date'])==keys and math.isclose(float(p['divCash']),float(issuer['amount']),rel_tol=0,abs_tol=1e-10)]
    return {'state':'PARTIAL_ACTUAL_EVIDENCE' if len(matches)==1 else 'BLOCKED_AMBIGUOUS_ACTION_MATCH','matches':len(matches),'P1_qualified':False}

def test_action_accounting(shares,open_price,dividend,split,cash,annual_cash_rate,days):
    """Pure TEST_ONLY conservation oracle, not real-data economics."""
    vals=[shares,open_price,dividend,split,cash,annual_cash_rate,days]
    if not all(math.isfinite(x) for x in vals) or shares<0 or open_price<=0 or dividend<0 or split<=0 or cash<0 or days<0:raise ValueError('Fixture contract')
    before=shares*open_price+cash;receivable=shares*dividend;price_ex=open_price-dividend
    if price_ex<=0:raise ValueError('Fixture ex price')
    shares2=shares*split;price2=price_ex/split;after_ex=shares2*price2+cash+receivable;paid_cash=cash+receivable;interest=paid_cash*annual_cash_rate/100*days/360
    return {'before':before,'after_ex':after_ex,'after_pay':shares2*price2+paid_cash,'interest':interest,'after_cash_accrual':shares2*price2+paid_cash+interest,'test_only':True,'real_PnL':False}
