"""Independent second pass: raw hashes/counts/arithmetic/SQLite replay/protection. No HTTP."""
import json,hashlib,csv,io,zipfile,sqlite3,math,urllib.robotparser
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from signalforge.runtime import file_hash,code_hash,validate_bundle,digest,now
from diagnostics_v3.audit import preserve
from data_recovery_v36.acquire import R,T,O,Q,load
from data_recovery_v36.runner import OT,write

def independent():
    acquired=load(O/'attempted_public_sources.json');plan=load(Q/'intent_register_v1.json');roster=load(Q/'official_roster_v1.json');assert plan['plan_id']==digest({k:v for k,v in plan.items() if k!='plan_id'});assert roster['roster_id']==digest({k:v for k,v in roster.items() if k!='roster_id'});assert file_hash(Q/'acquire.py')==load(O/'acquisition_code_binding.json')['acquire_code_sha256']
    attempts=acquired['attempts'];assert len(attempts)==acquired['requests']<=60 and len({a['url'] for a in attempts})==len(attempts);assert all(a['attempt']==1 for a in attempts);assert all(b['started_monotonic']-a['started_monotonic']>=1.049 for a,b in zip(attempts,attempts[1:]));assert sum(a.get('bytes',0) for a in attempts)==acquired['response_bytes']<250*1024*1024
    robots={};delays={}
    for a in attempts:
        if a.get('relative_bundle'):
            b=T/a['relative_bundle'];validate_bundle(b);payload=(b/'payload.bin').read_bytes();assert hashlib.sha256(payload).hexdigest()==a['sha256'] and len(payload)==a['bytes']<=12*1024*1024
        if a['kind']=='robots' and a.get('relative_bundle'):
            p=urllib.robotparser.RobotFileParser();p.parse((T/a['relative_bundle']/'payload.bin').read_text().splitlines());robots[a['url'].split('/')[2]]=p
            delays[a['url']]=p.crawl_delay('SignalForge-QX-v36-public-evidence/1.0')
    assert all(x is None or x<=1.05 for x in delays.values()),'Unmet robots delay; fail closed'
    public=[x for x in acquired['records'] if x.get('payload_path')]
    for x in public:
        b=Path(x['payload_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==x['sha256'];assert not x['first_public_qualified']
    assert len(public)==23;actual_claims=load(O/'document_qualification.json')['results'];assert len(actual_claims)==23 and not any(x['state']=='QUALIFIED_TIER_A' for x in actual_claims)
    rates=[];cftc_count=0;eia_count=0;independent_eia_bad=[]
    for x in public:
        b=Path(x['payload_path']).read_bytes()
        if x['family']=='CASH_NYFED':
            rows=json.loads(b)['refRates'];assert len({(z['type'],z['effectiveDate']) for z in rows})==len(rows)
            for z in rows:assert z['effectiveDate']<'2024-01-01' and z['effectiveDate'].startswith('2023-') and math.isfinite(z['percentRate']);rates.append((z['type'],z['effectiveDate'],float(z['percentRate'])))
        elif x['family']=='CFTC':
            z=zipfile.ZipFile(io.BytesIO(b));reader=list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode())));assert all(y['Report_Date_as_YYYY-MM-DD'].startswith('2023-') for y in reader);assert len({(y['CFTC_Contract_Market_Code'],y['Report_Date_as_YYYY-MM-DD']) for y in reader})==len(reader);cftc_count+=len(reader)
        elif 'table4.csv' in x['url']:
            values=list(csv.reader(io.StringIO(b.decode())));wanted={'Commercial (Excluding SPR)','Cushing','SPR','Total Motor Gasoline','Distillate Fuel Oil'}
            for row in values[1:]:
                if row[0] in wanted:
                    eia_count+=1;a,c,d=[float(v.replace(',','')) for v in row[1:4]]
                    if abs(a-c-d)>.0021:independent_eia_bad.append({'url':x['url'],'field':row[0],'error':a-c-d})
    assert len(rates)==499 and cftc_count==15343 and eia_count==20 and len(independent_eia_bad)==3
    observations=pd.read_csv(O/'public_observation_versions.csv');assert len(observations)==cftc_count+len(rates)+eia_count==15862 and not observations.first_public_qualified.any();assert not observations.duplicated(['source','entity','field','reference_date','payload_sha256']).any()
    actions=pd.read_csv(R/'reports/p1_pit_v34/available_action_rows.csv');matches=pd.read_csv(O/'issuer_actions_delta.csv');assert len(actions)==438 and actions.pay_date.isna().all() and len(matches)==24
    for x in matches.itertuples():
        g=actions[(actions.asset==x.asset)&(actions.ex_date==x.ex_date)];assert len(g)==1 and abs(float(g.iloc[0].divCash)-x.amount)<1e-10 and x.pay_date>=x.ex_date and x.pay_date<'2024-01-01';assert x.matches==1 and not x.P1_qualified
    rawissuer=load(R/'reports/p1_pit_v34/issuer_action_reconciliation.json');assert set(zip(matches.asset,matches.ex_date,matches.amount))=={(x['asset'],x['normalized_ex_date'],x['distribution_per_share']) for x in rawissuer['rows']}
    sessions=pd.read_csv(O/'p1_scope_gaps.csv');assert len(sessions)==27088 and not sessions.duplicated(['asset','date']).any() and (sessions.date<'2024-01-01').all();assert not sessions.P1_eligible.any() and not sessions.asof_cash_qualified.any();assert int(sessions.cash_documentary_date_present.sum())==1992 and sessions.issuer_pay_date_partial.notna().sum()==24
    canonical=load(O/'genuine_certification_receipt.json');assert canonical['prior_aggregate_versions']==67448 and canonical['after_Tier_A']==canonical['new_P1_sessions']==0
    shadow=load(O/'shadow_readiness_receipt_v2.json');dbpath=Path(shadow['database']);shadowplan=load(O/'shadow_prepared_plan_v2.json');assert shadowplan['plan_id']==digest({k:v for k,v in shadowplan.items() if k!='plan_id'})
    with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:rows=db.execute('SELECT id,retry_key,semantic_identity,message_sha256,observed_utc,body,previous_sha256,receipt_sha256 FROM records ORDER BY id').fetchall()
    canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode();h=lambda b:hashlib.sha256(b).hexdigest();prev='0'*64;last=None;identities=set()
    for expected,row in enumerate(rows,1):
        i,key,identity,mhash,at,body,prior,receipt=row;m=json.loads(body);d=datetime.fromisoformat(m['decision_at'].replace('Z','+00:00')).astimezone(timezone.utc);instant=datetime.fromisoformat(at);assert i==expected and prior==prev and h(body.encode())==mhash and m['kind']=='TEST_ONLY';assert last is None or instant>=last;assert identity==h(canon([m['track'],d.isoformat(),m['asset'],m['seed']])) and identity not in identities;identities.add(identity)
        bound=shadowplan['track_bindings'][m['track']]
        for k in ['model_hash','source_hash','normalizer_hash','target_hash','benchmark_hash']:assert m[k] in bound[k+'s']
        c={'id':i,'retry_key':key,'semantic_identity':identity,'message_sha256':mhash,'observed_utc':at,'previous_sha256':prior,'state':'TEST_ONLY_READINESS_NOT_FORECAST'};assert h(canon(c))==receipt;prev=receipt;last=instant
    assert len(rows)==48 and prev==shadow['replay']['head_sha256']
    pre=load(O/'preflight.json')
    for path,expected in pre['protected_files'].items():assert file_hash(R/path)==expected,path
    assert len(pre['protected_files'])==468
    p35=load(R/'reports/qualification_v35/preflight.json')
    for path,expected in p35['protected_files'].items():assert file_hash(R/path)==expected,path
    idx=load(R/'reports/calibration_v34/final_index.json');validate_bundle(T/idx['relative_bundle'])
    for path,expected in idx['files'].items():assert file_hash(R/path)==expected,path
    fits=load(R/'reports/calibration_v34/oof_completion.json')['results']
    for fit in fits:validate_bundle(T/fit['relative_bundle']);assert file_hash(fit['checkpoint_path'])==fit['checkpoint_hash']
    assert len(fits)==120
    p=preserve();assert code_hash(R)=='471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870';assert code_hash(Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX'))=='7cdf374c83afd29e07c79e2ea8a0cc10ff2cad3637cee6ff01fa95ba9f7049bf'
    tests=load(O/'tests.json');assert tests['exit_code']==0 and tests['failures']==tests['errors']==tests['skips']==0
    result={'passed':True,'observed_at':now(),'v36_code_hash':digest({str(x.relative_to(Q)):file_hash(x) for x in Q.glob('*.py')}),'intent_plan_id':plan['plan_id'],'roster_id':roster['roster_id'],'requests':len(attempts),'publisher_content_documents':len(public),'new_payload_documents':acquired['new_payload_documents'],'independent_observation_versions':len(observations),'CFTC_archive_rows':cftc_count,'cash_rates':len(rates),'EIA_csv_rows':eia_count,'EIA_csv_arithmetic_failures':independent_eia_bad,'new_Tier_A':0,'new_P1':0,'issuer_unique_partial':24,'new_issuer_unique_partial':0,'P1_session_denominator':27088,'asset_years':sessions.groupby(['asset','year']).ngroups,'cash_documentary_asset_sessions':1992,'shadow_test_records':48,'actual_forecasts':0,'actual_future_dates':0,'original_files_verified':357,'protected_dev_files_verified':468,'v35_prior_protected_verified':len(p35['protected_files']),'v34_models_checkpoints_verified':120,'v34_report_files_verified':len(idx['files']),'frozen_v2_hash':code_hash(R),'preservation':p,'tests':tests,'reserved_access':False,'GPU_training':0,'P1_PnL':False,'original_first_public_authenticated':0,'canonical_mixed_clock_export_read':False,'robots_delays':delays,'historical_v1_hash_test_failure_preserved':True};write('independent_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['preservation','tests','EIA_csv_arithmetic_failures']}),flush=True);return result
if __name__=='__main__':independent()
