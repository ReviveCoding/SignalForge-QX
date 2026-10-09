"""Bounded Tiingo corporate-action preparation; never self-qualifies P1 economics."""
import json,math,os,re
from pathlib import Path
from urllib.parse import urlsplit,parse_qsl,urlencode
import pandas as pd
from .runtime import digest,file_hash,now,atomic_json,commit_bundle,validate_bundle
from .sources import Acquisition
from .tiingo import SYMBOLS,START,END

ACTION_TYPES=('distributions','splits')


def new_plan():
    return {'source':'tiingo_corporate_actions','partition':'development','reserved_access':False,
        'registered_at':now(),'request_selection':'outcome_blind','symbols':SYMBOLS,'action_types':list(ACTION_TYPES),
        'startExDate':START,'endExDate':END,'max_bytes':16*1024**2,'max_bytes_per_request':1024**2,
        'max_rows_per_request':5000,'economic_qualified':False}


def validate_plan(plan):
    expected=set(new_plan())
    if set(plan)!=expected:raise ValueError('Exact secret-free Tiingo action plan fields required')
    reference=new_plan()
    for key,value in reference.items():
        if key in {'registered_at','max_bytes','max_bytes_per_request','max_rows_per_request'}:continue
        if type(plan[key]) is not type(value) or plan[key]!=value:raise PermissionError('Frozen Tiingo action contract differs: '+key)
    if not 0<plan['max_bytes']<=16*1024**2 or not 0<plan['max_bytes_per_request']<=1024**2 or not 0<plan['max_rows_per_request']<=5000:
        raise ValueError('Tiingo action acquisition ceiling')
    t=pd.Timestamp(plan['registered_at'])
    if t.tzinfo is None or t>pd.Timestamp.now(tz='UTC'):raise ValueError('Aware action-plan registration required')
    return digest(plan)


def register_plan(repo,runtime):
    path=Path(repo)/'.local/tiingo_action_plan.json'
    if path.exists():plan=json.loads(path.read_text(encoding='utf-8-sig'))
    else:plan=new_plan();atomic_json(path,plan)
    identity=validate_plan(plan)
    commit_bundle(Path(runtime)/'artifacts/tiingo_action_plans'/identity,{'plan.json':plan},{'reserved_access':False})
    return plan


def validate_action_url(url):
    u=urlsplit(url);pairs=parse_qsl(u.query,keep_blank_values=True);params=dict(pairs)
    if u.scheme!='https' or u.hostname!='api.tiingo.com' or u.port not in (None,443) or u.username or u.password or u.fragment or len(pairs)!=len(params):
        raise PermissionError('Exact Tiingo corporate-action HTTPS URL required')
    match=re.fullmatch(r'/tiingo/corporate-actions/([A-Z]+)/(distributions|splits)',u.path)
    if not match or match[1] not in SYMBOLS:raise PermissionError('Frozen Tiingo action ticker required')
    if params!={'startExDate':START,'endExDate':END}:raise PermissionError('Exact bounded action date range required')
    return match[1],match[2]


def request(symbol,kind,plan):
    validate_plan(plan)
    if symbol not in SYMBOLS or kind not in ACTION_TYPES:raise PermissionError('Frozen action request required')
    url=f'https://api.tiingo.com/tiingo/corporate-actions/{symbol}/{kind}'
    params={'startExDate':START,'endExDate':END}
    validate_action_url(url+'?'+urlencode(params));return url,params


def _date(value,name,required=False):
    if value in (None,''):
        if required:raise ValueError(name+' required')
        return None
    t=pd.Timestamp(value)
    if t.tzinfo is None:t=t.tz_localize('UTC')
    else:t=t.tz_convert('UTC')
    if not pd.Timestamp(START,tz='UTC')<=t<pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError(name+' outside development bounds')
    return t.isoformat()


def parse_actions(path,symbol,kind,plan=None,expected_hash=None):
    plan=plan or new_plan();validate_plan(plan);path=Path(path)
    if path.stat().st_size>plan['max_bytes_per_request']:raise ValueError('Tiingo action response byte ceiling')
    sha=file_hash(path)
    if expected_hash and sha!=expected_hash:raise ValueError('Immutable action checksum mismatch')
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or len(rows)>plan['max_rows_per_request']:raise ValueError('Bounded action array required')
    parsed=[]
    for row in rows:
        if not isinstance(row,dict) or row.get('ticker')!=symbol:raise ValueError('Action ticker mismatch')
        if kind=='distributions':
            if 'exDate' not in row or 'distribution' not in row:raise ValueError('Distribution exDate/amount required')
            amount=row['distribution']
            if type(amount) not in (int,float) or not math.isfinite(amount) or amount<0:raise ValueError('Finite nonnegative distribution required')
            parsed.append({'asset':symbol,'kind':'distribution','exDate':_date(row.get('exDate'),'exDate',True),
                'paymentDate':_date(row.get('paymentDate'),'paymentDate'),'recordDate':_date(row.get('recordDate'),'recordDate'),
                'declarationDate':_date(row.get('declarationDate'),'declarationDate'),'distribution':float(amount),
                'distributionFrequency':row.get('distributionFrequency'),'raw_hash':sha})
        else:
            required={'exDate','splitFrom','splitTo','splitFactor','splitStatus'}
            if not required<=set(row):raise ValueError('Split fields required')
            values=[row['splitFrom'],row['splitTo'],row['splitFactor']]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive finite split ratios required')
            if row['splitStatus'] not in {'a','c'}:raise ValueError('Known split status required')
            parsed.append({'asset':symbol,'kind':'split','exDate':_date(row['exDate'],'exDate',True),
                'splitFrom':float(row['splitFrom']),'splitTo':float(row['splitTo']),'splitFactor':float(row['splitFactor']),
                'splitStatus':row['splitStatus'],'raw_hash':sha})
    return parsed


def acquire_plan(repo,runtime,plan,client=None):
    repo,runtime=Path(repo),Path(runtime);plan_id=validate_plan(plan)
    if not os.environ.get('TIINGO_API_TOKEN'):raise PermissionError('BLOCKED_AUTH: TIINGO_API_TOKEN absent')
    client=client or Acquisition(runtime,plan['max_bytes']);records=[];raws=[];total=0
    for symbol in SYMBOLS:
        for kind in ACTION_TYPES:
            url,params=request(symbol,kind,plan);raw=client.cached(url,'tiingo_action',params)
            if raw is None:
                saved=getattr(client,'remaining',None)
                if saved is not None:client.remaining=min(plan['max_bytes_per_request'],plan['max_bytes']-total)
                raw=client.fetch(url,'tiingo_action',params,{'symbol':symbol,'action_type':kind,'request_plan_id':plan_id,'reserved_access':False})
                if saved is not None:client.remaining=saved-raw['bytes']
            total+=Path(raw['path']).stat().st_size
            if total>plan['max_bytes']:raise RuntimeError('Tiingo action total byte ceiling reached')
            records.extend(parse_actions(raw['path'],symbol,kind,plan,raw['sha256']));raws.append(raw)
    complete_pay_dates=all(r['kind']!='distribution' or r['paymentDate'] is not None for r in records)
    result={'state':'SUCCEEDED_RAW_TIINGO_ACTION_DEVELOPMENT','source':'tiingo_corporate_actions','plan_id':plan_id,
        'records':records,'raw_receipts':raws,'complete_payment_dates':complete_pay_dates,'reserved_access':False,
        'pit_tier':'B','economic_qualified':False,'qualified_for_final':False,'created_at':now(),
        'claim_boundary':'Corporate-action preparation only; provider entitlement, historical original publication clocks, action completeness and P1 accounting qualification remain separate'}
    identity=digest(result);commit_bundle(runtime/'artifacts/tiingo_action_results'/identity,{'results.json':result},{'plan_id':plan_id})
    atomic_json(repo/'reports/tiingo_action_acquisition.json',{'state':result['state'],'artifact_id':identity,'plan_id':plan_id,
        'rows':len(records),'raw_receipts':len(raws),'complete_payment_dates':complete_pay_dates,'economic_qualified':False,'reserved_access':False})
    return result
