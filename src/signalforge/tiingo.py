"""Bounded EOD development ingestion; reconstructed clocks never qualify final/P1."""
import json, math, os, re
import datetime
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode
import pandas as pd
import numpy as np
from .runtime import digest, file_hash, now, atomic_json, commit_bundle, validate_bundle
from .sources import Acquisition

SYMBOLS = ['SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG']
START, END = '2010-07-20', '2023-12-31'
FIELDS = ['date','open','high','low','close','volume','adjOpen','adjHigh','adjLow','adjClose','adjVolume','divCash','splitFactor']
RAW_FIELDS = {'date','open','high','low','close','volume','divCash','splitFactor'}
ADJUSTED_FIELDS = set(FIELDS)-RAW_FIELDS
UNITS = {'return_1w':'fraction','return_4w':'fraction','realized_vol_20':'daily_fraction','log_volume':'log1p_shares'}
CLOCK = 'reconstructed_Tiingo_EOD_available_by_20_30_America_New_York_not_original_publication'
EOD_RECONSTRUCTED_AVAILABLE_LOCAL_TIME = datetime.time(20,30)


def reconstructed_eod_available_at(close):
    """Conservative development replay bound, not authenticated historical publication time."""
    stamp=pd.Timestamp(close)
    if stamp.tzinfo is None:raise ValueError('Aware exchange close required')
    local=stamp.tz_convert('America/New_York')
    return pd.Timestamp(datetime.datetime.combine(local.date(),EOD_RECONSTRUCTED_AVAILABLE_LOCAL_TIME),tz='America/New_York').tz_convert('UTC')


def new_plan():
    return {'source':'tiingo','partition':'development','reserved_access':False,
        'registered_at':now(),'request_selection':'outcome_blind','symbols':SYMBOLS,
        'startDate':START,'endDate':END,'resampleFreq':'daily','format':'json',
        'max_bytes':16*1024**2,'max_bytes_per_symbol':2*1024**2,'max_pages_per_symbol':1,
        'max_rows_per_symbol':4000,'fields':FIELDS,'calendar':'XNYS',
        'price_mode':'P0','basis':'unadjusted_close_price_return'}


def validate_plan(plan):
    expected=set(new_plan())
    if set(plan)!=expected:raise ValueError('Exact secret-free Tiingo plan fields required')
    for key,value in new_plan().items():
        if key in {'registered_at','max_bytes','max_bytes_per_symbol','max_rows_per_symbol'}:continue
        if type(plan[key]) is not type(value) or plan[key]!=value:
            raise PermissionError('Frozen Tiingo development request contract differs: '+key)
    for key,ceiling in [('max_bytes',16*1024**2),('max_bytes_per_symbol',2*1024**2),('max_rows_per_symbol',4000)]:
        if type(plan[key]) is not int or not 0<plan[key]<=ceiling:raise ValueError('Tiingo budget ceiling: '+key)
    t=pd.Timestamp(plan['registered_at'])
    if t.tzinfo is None or t>pd.Timestamp.now(tz='UTC'):raise ValueError('Actual aware plan registration required')
    return digest(plan)


def register_plan(repo,runtime):
    path=Path(repo)/'.local/tiingo_price_plan.json'
    if path.exists():plan=json.loads(path.read_text(encoding='utf-8-sig'))
    else:
        plan=new_plan();atomic_json(path,plan)
    identity=validate_plan(plan)
    commit_bundle(Path(runtime)/'artifacts/tiingo_request_plans'/identity,{'plan.json':plan},{'reserved_access':False})
    return plan


def validate_tiingo_url(url):
    u=urlsplit(url);pairs=parse_qsl(u.query,keep_blank_values=True);params=dict(pairs)
    if (u.scheme!='https' or u.hostname!='api.tiingo.com' or u.port not in (None,443) or
        u.username or u.password or u.fragment or len(pairs)!=len(params)):
        raise PermissionError('Exact Tiingo HTTPS host and unique bounded parameters required')
    match=re.fullmatch(r'/tiingo/daily/([A-Z]+)/prices',u.path)
    if not match or match[1] not in SYMBOLS:raise PermissionError('No Tiingo current/metadata endpoint permitted')
    if params!={'startDate':START,'endDate':END,'resampleFreq':'daily','format':'json'}:
        raise PermissionError('Explicit exact Tiingo development date bounds required; no token query')
    return match[1]


def request(symbol,plan):
    validate_plan(plan)
    if symbol not in SYMBOLS:raise PermissionError('Frozen Tiingo symbol required')
    params={k:plan[k] for k in ['startDate','endDate','resampleFreq','format']}
    url='https://api.tiingo.com/tiingo/daily/'+symbol+'/prices'
    validate_tiingo_url(url+'?'+urlencode(params))
    return url,params


def parse_tiingo(path,symbol,plan=None,expected_hash=None):
    plan=plan or new_plan();validate_plan(plan)
    if symbol not in SYMBOLS:raise ValueError('Frozen symbol required')
    path=Path(path)
    if path.stat().st_size>plan['max_bytes_per_symbol']:raise ValueError('Tiingo symbol byte ceiling')
    sha=file_hash(path)
    if expected_hash is not None and sha!=expected_hash:raise ValueError('Immutable Tiingo raw checksum mismatch')
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or not 0<len(rows)<=plan['max_rows_per_symbol']:
        raise ValueError('Nonempty bounded Tiingo EOD array required')
    parsed=[];previous=None
    for row in rows:
        if not isinstance(row,dict) or not RAW_FIELDS<=set(row) or not set(row)<=set(FIELDS):
            raise ValueError('Required raw EOD schema with optional adjusted fields; metadata forbidden')
        if not isinstance(row['date'],str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T00:00:00(?:\.000)?(?:Z|\+00:00)',row['date']):
            raise ValueError('Tiingo UTC session-date label required')
        day=pd.Timestamp(row['date']).date().isoformat()
        if not START<=day<=END:raise PermissionError('Tiingo response outside pre-2024 development bounds')
        if previous is not None and day<=previous:raise ValueError('Strict monotonic unique Tiingo dates required')
        previous=day
        for field in set(row)-{'date'}:
            value=row[field]
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative Tiingo numeric fields required')
        if row['splitFactor']<=0:raise ValueError('Positive splitFactor required')
        for prefix in ['','adj']:
            names=['open','high','low','close'] if not prefix else ['adjOpen','adjHigh','adjLow','adjClose']
            if not set(names)<=set(row):continue
            op,hi,lo,cl=[row[k] for k in names]
            if not lo<=min(op,cl)<=max(op,cl)<=hi:raise ValueError('Tiingo OHLC ordering invalid')
        parsed.append({**row,'session_date':day,'asset':symbol,'raw_hash':sha,
            'raw_price_basis':'unadjusted','adjusted_fields_present':sorted(set(row)&ADJUSTED_FIELDS)})
    if file_hash(path)!=sha:raise ValueError('Tiingo raw changed during parsing')
    return parsed


def acquire_plan(repo,runtime,plan,client=None):
    plan_id=validate_plan(plan)
    commit_bundle(Path(runtime)/'artifacts/tiingo_request_plans'/plan_id,{'plan.json':plan},{'reserved_access':False})
    if not os.environ.get('TIINGO_API_TOKEN'):raise PermissionError('BLOCKED_AUTH: TIINGO_API_TOKEN absent')
    client=client or Acquisition(runtime,plan['max_bytes']);raws=[];records=[];total=0
    for symbol in SYMBOLS:
        url,params=request(symbol,plan)
        raw=client.cached(url,'tiingo',params)
        if raw is None:
            # Bound each response during streaming, not only after downloading it.
            remaining=getattr(client,'remaining',None)
            if remaining is not None:client.remaining=min(plan['max_bytes_per_symbol'],plan['max_bytes']-total)
            raw=client.fetch(url,'tiingo',params,{'symbol':symbol,'request_plan_id':plan_id,'reserved_access':False})
            if remaining is not None:client.remaining=remaining-raw['bytes']
        size=Path(raw['path']).stat().st_size;total+=size
        if total>plan['max_bytes']:raise RuntimeError('Tiingo total byte ceiling reached')
        records.extend(parse_tiingo(raw['path'],symbol,plan,raw['sha256']));raws.append(raw)
    result={'state':'SUCCEEDED_RAW_TIINGO_DEVELOPMENT','source':'tiingo','plan_id':plan_id,
        'records':records,'raw_receipts':raws,'pit_tier':'B','evidence_kind':'public_data_reconstructed',
        'original_publication_qualified':False,'qualified_for_final':False,'economic_qualified':False,
        'reserved_access':False,'created_at':now()}
    identity=digest(result)
    commit_bundle(Path(runtime)/'artifacts/tiingo_request_results'/identity,{'results.json':result},{'plan_id':plan_id})
    result['artifact_id']=identity
    # Large raw/parsed arrays stay on ext4; the Windows report is a small pointer.
    summary={k:v for k,v in result.items() if k not in {'records','raw_receipts'}}
    summary.update(rows=len(records),raw_receipts=raws,relative_path='artifacts/tiingo_request_results/'+identity+'/results.json')
    atomic_json(Path(repo)/'reports/tiingo_development_acquisition.json',summary)
    return summary


def canonical_inputs(records,origins):
    """Fixtures may call this pure transform; public manifests require immutable live receipts."""
    import exchange_calendars as xc
    cal=xc.get_calendar('XNYS',start=START,end=END)
    schedule={str(index.date()):row['close'] for index,row in cal.schedule.iterrows()}
    closes=pd.DatetimeIndex(list(schedule.values()))
    conservative_available={str(close.date()):reconstructed_eod_available_at(close) for close in closes}
    availability_index=pd.DatetimeIndex([conservative_available[str(close.date())] for close in closes])
    stamps=[pd.Timestamp(origin) for origin in origins]
    if any(t.tzinfo is None for t in stamps):raise ValueError('Aware development origins required')
    if any(t<pd.Timestamp(START,tz='UTC') or t>=pd.Timestamp('2024-01-01T00:00Z') for t in stamps):
        raise PermissionError('Development origins outside frozen bounds')
    if any(t.tz_convert('America/New_York').weekday()!=4 or
           t.tz_convert('America/New_York').time().isoformat()!='18:00:00' for t in stamps):
        raise ValueError('Friday 18:00 America/New_York required')
    if any(right-left!=pd.Timedelta(weeks=1) for left,right in
           zip([t.tz_convert('America/New_York').tz_localize(None) for t in stamps],
               [t.tz_convert('America/New_York').tz_localize(None) for t in stamps][1:])):
        raise ValueError('Contiguous weekly grid required; no favorable intersection')
    if any(r['asset'] not in SYMBOLS for r in records):raise ValueError('Unknown market symbol')
    marks=[];events=[];daily=[];actions=[];coverage=[]
    for asset in SYMBOLS:
        rows=sorted([r for r in records if r['asset']==asset],key=lambda r:r['session_date'])
        by_date={r['session_date']:r for r in rows}
        if len(by_date)!=len(rows):raise ValueError('Duplicate symbol sessions')
        if len({r['raw_hash'] for r in rows})>1:raise ValueError('One immutable response per symbol required')
        raw_hash=rows[0]['raw_hash'] if rows else None
        # Reindex to sessions so a missing session cannot masquerade as a 20-session window.
        frame=pd.DataFrame([{'day':day,'close':by_date[day]['close'] if day in by_date else np.nan,
            'volume':by_date[day]['volume'] if day in by_date else np.nan} for day in schedule])
        returns=frame.close.pct_change(fill_method=None)
        volatility=returns.rolling(20,min_periods=20).std(ddof=1)
        weekly=[]
        for origin in origins:
            t=pd.Timestamp(origin)
            if t.tzinfo is None or t>=pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError('Aware development origins required')
            idx=availability_index.searchsorted(t.tz_convert('UTC'),side='right')-1
            if idx<0:raise ValueError('No conservatively available EOD mark before registered origin')
            close=closes[idx];day=close.date().isoformat();r=by_date.get(day);available=conservative_available[day]
            weekly.append(r['close'] if r is not None and r['close']>0 else np.nan)
            if r is not None:
                marks.append({'asset':asset,'decision_time':t.isoformat(),'time':close.isoformat(),'available_at':available.isoformat(),
                    'close':r['close'],'fresh':True,'adjusted':False,'raw_hash':r['raw_hash'],
                    'pit_tier':'B','clock_evidence':CLOCK,'original_publication_qualified':False})
            position=frame.index[frame.day.eq(day)][0]
            values={'return_1w':weekly[-1]/weekly[-2]-1 if len(weekly)>1 and np.isfinite(weekly[-2]) else np.nan,
                'return_4w':weekly[-1]/weekly[-5]-1 if len(weekly)>4 and np.isfinite(weekly[-5]) else np.nan,
                'realized_vol_20':volatility.iloc[position],
                'log_volume':math.log1p(r['volume']) if r is not None else np.nan}
            for field,value in values.items():
                if raw_hash is None:continue
                parents=[raw_hash] # Each symbol response binds the whole bounded history.
                events.append({'entity':asset,'source':'market','field':field,'value':float(value) if np.isfinite(value) else None,'unit':UNITS[field],
                    'reference_time':close.isoformat(),'available_at':available.isoformat(),'raw_hash':raw_hash,
                    'dependency_raw_hashes':parents,'pit_tier':'B','clock_evidence':CLOCK,'original_publication_qualified':False})
        for r in rows:
            if r['session_date'] not in schedule:raise ValueError('Tiingo row outside XNYS regular sessions')
            close=schedule[r['session_date']]
            daily.append({'asset':asset,'session_date':r['session_date'],'raw_open':r['open'],'raw_close':r['close'],
                'divCash':r['divCash'],'splitFactor':r['splitFactor'],'raw_hash':r['raw_hash'],
                'available_at':reconstructed_eod_available_at(close).isoformat(),'pit_tier':'B','clock_evidence':CLOCK})
            if r['divCash'] or r['splitFactor']!=1:
                actions.append({'asset':asset,'ex_date':r['session_date'],'divCash':r['divCash'],
                    'splitFactor':r['splitFactor'],'pay_date':None,'raw_hash':r['raw_hash'],'audited':False})
        coverage.append({'asset':asset,'returned_sessions':len(rows),'expected_sessions':len(schedule),
            'missing_sessions':sorted(set(schedule)-set(by_date)),'action_coverage_qualified':False})
    return marks,events,{'daily':daily,'actions':actions,'coverage':coverage,'state':'BLOCKED_ACTION_CLOCK_AUDIT',
        'economic_qualified':False,'qualified_for_final':False,'pay_dates_available':False,
        'contract':{'price_basis':'raw_open_plus_actions_only','p0_substitution_permitted':False,
            'adjusted_price_plus_dividend_permitted':False,'required_qualification_evidence':[
                'provider_access_license','immutable_price_action_hashes','complete_session_action_coverage',
                'original_open_close_publication_clocks','ex_date_entitlement_and_pay_dates',
                'split_share_units','self_financing_accounting_oracles'],
            'admission':'separate explicit hash-bound action/clock audit; this preparation never grants admission'},
        'reason':'EOD ex-date fields do not authenticate action completeness, pay-date entitlements or original open/final close publication clocks; explicit audit required'}


def build_inputs(repo,runtime):
    repo,runtime=Path(repo),Path(runtime);plan=register_plan(repo,runtime);plan_id=validate_plan(plan)
    summary_path=repo/'reports/tiingo_development_acquisition.json'
    if not summary_path.exists():raise RuntimeError('BLOCKED_DATA: real bounded Tiingo receipt absent; no public cards created')
    summary=json.loads(summary_path.read_text())
    if summary.get('plan_id')!=plan_id or summary.get('state')!='SUCCEEDED_RAW_TIINGO_DEVELOPMENT':raise ValueError('Matching real acquisition required')
    if not re.fullmatch('[a-f0-9]{64}',summary.get('artifact_id','')):raise ValueError('Invalid result identity')
    directory=runtime/'artifacts/tiingo_request_results'/summary['artifact_id'];validate_bundle(directory)
    result=json.loads((directory/'results.json').read_text());records=[]
    if (digest(result)!=summary['artifact_id'] or result.get('plan_id')!=plan_id or
        result.get('state')!='SUCCEEDED_RAW_TIINGO_DEVELOPMENT' or result.get('reserved_access') is not False or
        result.get('evidence_kind')!='public_data_reconstructed' or len(result['raw_receipts'])!=8):
        raise PermissionError('Real frozen-universe result identity and receipts required')
    for symbol,raw in zip(SYMBOLS,result['raw_receipts']):
        url,params=request(symbol,plan)
        from .sources import request_identity
        expected=runtime/'raw'/request_identity(url,params)
        receipt_path=expected/(raw['sha256']+'.receipt.json')
        if (not Path(raw['path']).resolve().is_relative_to(expected.resolve()) or raw.get('evidence_kind')!='real_pilot' or
            raw.get('metadata',{}).get('request_plan_id')!=plan_id or raw.get('metadata',{}).get('symbol')!=symbol or
            not receipt_path.is_file() or json.loads(receipt_path.read_text())!=raw or
            validate_tiingo_url(raw.get('source_url',''))!=symbol):raise PermissionError('Live plan-bound raw receipt required')
        records.extend(parse_tiingo(raw['path'],symbol,plan,raw['sha256']))
    if records!=result['records']:raise ValueError('Parsed result differs from immutable raw lineage')
    if sum(Path(r['path']).stat().st_size for r in result['raw_receipts'])>plan['max_bytes']:
        raise ValueError('Total Tiingo byte ceiling')
    manifests=[]
    for track,start in [('Main-A',START),('Nested-B','2019-10-01')]:
        # DST-safe civil Friday clock, retaining holidays and missing marks.
        origins=[(d.tz_localize('America/New_York')+pd.Timedelta(hours=18)).isoformat()
            for d in pd.date_range(start,END,freq='W-FRI')]
        prices,events,p1=canonical_inputs(records,origins)
        identity=digest({'plan_id':plan_id,'track':track,'raw_hashes':[r['sha256'] for r in result['raw_receipts']],
            'builder_version':'tiingo_p0_v2_conservative_2030','builder_hash':file_hash(Path(__file__))})
        folder=runtime/'artifacts/tiingo_inputs'/identity
        commit_bundle(folder,{'decisions.json':origins,'prices.json':prices,'events.json':events,'p1_preparation.json':p1},
            {'track':track,'plan_id':plan_id,'qualified_for_final':False,'economic_qualified':False})
        def card(name):return {'relative_path':str((folder/name).relative_to(runtime)),'sha256':file_hash(folder/name)}
        info=['I0','I1','I2','I3']+(['I4'] if track=='Nested-B' else [])
        features=[{'name':k,'source':'market','field':k,'entity':'ASSET','unit':v} for k,v in UNITS.items()]
        manifest={'schema':'release_aware_track_inputs_v1','reserved_access':False,'development_end':END,
            'evidence_kind':'public_data_reconstructed','qualified_for_final':False,'economic_qualified':False,
            'price_mode':'P0','p0_basis':'unadjusted_close_price_return','decisions':card('decisions.json'),
            'universe_card':{'assets':SYMBOLS,'outcome_blind':True,'frozen_before_outcomes':plan_id},
            'information_sets':info,'inputs':{'prices':card('prices.json'),'events':card('events.json')},
            'features':features,'plan_id':plan_id,'calendar':'XNYS','clock_evidence':CLOCK,
            'original_publication_qualified':False,'p1_preparation':card('p1_preparation.json')}
        path=repo/'.local'/('inputs_'+track+'.json')
        if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Existing track manifest preserved; explicit reconciliation required')
        atomic_json(path,manifest);manifests.append({'track':track,'manifest_id':digest(manifest),'grid_rows':len(origins)*8})
    report={'state':'READY_P0_I0_CPU_DEVELOPMENT','tracks':manifests,'pit_tier':'B','qualified_for_final':False,
        'economic_qualified':False,'reserved_access':False,'p1_state':'BLOCKED_ACTION_CLOCK_AUDIT','plan_id':plan_id}
    atomic_json(repo/'reports/tiingo_input_build.json',report);return report
