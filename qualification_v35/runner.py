"""Execute bounded V35 official evidence acquisition, qualification and readiness."""
import os,sys,json,subprocess,urllib.request,urllib.error,re,hashlib,csv,math
from pathlib import Path
import pandas as pd,numpy as np
from signalforge.runtime import file_hash,code_hash,digest,now,validate_bundle,commit_bundle
from qualification_v35.core import *
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering');ORIG=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX');OT=Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1');O=R/'reports/qualification_v35';Q=R/'qualification_v35'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,data):
    path=O/name
    if path.exists():
        assert load(path)==data,'Immutable receipt differs: '+name
        return
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def emit(stage,**kw):print(json.dumps({'at':now(),'stage':stage,'GPU_training':0,'reserved_access':False,**kw}),flush=True)
def hashcode():return digest({str(p.relative_to(R)):file_hash(p) for p in sorted(Q.glob('*.py'))})
class RedirectScope(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        from urllib.parse import urlparse
        if urlparse(newurl).hostname!=urlparse(req.full_url).hostname or urlparse(newurl).scheme!='https':raise PermissionError('Public source redirect outside fixed publisher scope')
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def acquire(plan):
    existing=O/'verified_publisher_documents.json'
    if existing.exists():
        records=load(existing)['records']
        for r in records:
            if r.get('relative_bundle'):validate_bundle(T/r['relative_bundle'])
        return records
    records=[];opener=urllib.request.build_opener(RedirectScope())
    for i,url in enumerate(plan['allowed_urls']):
        emit('OFFICIAL_DOCUMENT_REQUEST',request=i+1,maximum=plan['maximum_requests'],url=url)
        row={'url':url,'requested_at':now(),'first_public_certified':False,'scope':'2023 issuer/EIA historical payload or source semantics documentation only','attempt':1}
        try:
            with opener.open(urllib.request.Request(url,headers={'User-Agent':'SignalForge-QX public-evidence-qualification/3.5'}),timeout=plan['request_timeout_seconds']) as response:
                payload=response.read(plan['maximum_response_bytes']+1);assert len(payload)<=plan['maximum_response_bytes'];row.update(status=response.status,final_url=response.geturl(),headers={k:response.headers.get(k) for k in ['Content-Type','Last-Modified','ETag']})
            assert payload.startswith(b'%PDF') if url.endswith('.pdf') else b'<html' in payload.lower()[:3000]
            row.update(state='VERIFIED_PAYLOAD_BUT_NO_FIRST_PUBLIC',sha256=sha(payload),bytes=len(payload),retrieved_at=now(),headers_are_not_historical_clock=True)
            ext='document.pdf' if url.endswith('.pdf') else 'document.html';folder=T/'artifacts/qualification_v35/public'/digest(row);commit_bundle(folder,{ext:payload,'retrieval.json':row},{'evidence_only':True,'reserved_access':False});row.update(relative_bundle=str(folder.relative_to(T)),filename=ext)
        except Exception as error:
            row.update(state='PUBLIC_RETRIEVAL_FAILED',error_type=type(error).__name__,http_status=getattr(error,'code',None),no_retry=True,no_gate_promoted=True)
        records.append(row);emit('OFFICIAL_DOCUMENT_RESULT',state=row['state'],http_status=row.get('http_status',row.get('status')))
    write('verified_publisher_documents.json',{'plan_id':plan['plan_id'],'records':records,'maximum_requests':5,'credential_requests':0,'private_403_retries':0})
    return records

def issuer_delta(records):
    prior=load(R/'reports/p1_pit_v34/issuer_action_reconciliation.json');issuer=load(ORIG/'reports/issuer_action_evidence_2023.json');parsed=[]
    doc=next((r for r in records if 'ishares.com' in r['url'] and r.get('relative_bundle')),None)
    if doc:
        pdf=T/doc['relative_bundle']/doc['filename'];assert file_hash(pdf)==doc['sha256']
        extraction='import fitz,json,sys; p=fitz.open(sys.argv[1]); print(json.dumps([{\"page\":i+1,\"text\":p[i].get_text()} for i in [15,27]]))'
        pages=json.loads(subprocess.check_output([str(OT/'tools/pdf_qa_env/bin/python'),'-c',extraction,str(pdf)],text=True,timeout=30))
        for page in pages:
            lines=page['text'].splitlines()
            for i,line in enumerate(lines):
                if line.strip() not in ['IEF','TLT']:continue
                seg=lines[i:i+12];dates=[x for x in seg if re.fullmatch(r'\d\d/\d\d/2023',x.strip())];assert len(dates)==3
                amount=next(x.strip() for x in seg[seg.index(dates[-1])+1:] if re.fullmatch(r'\d+\.\d+',x.strip()))
                parsed.append({'asset':line.strip(),'record_date':dates[0],'ex_date':dates[1],'pay_date':dates[2],'distribution_per_share':float(amount),'pdf_page':page['page']})
        assert parsed==issuer['rows'],'Fresh issuer payload differs; preserve conflict before qualification'
    else:parsed=issuer['rows'] # immutable prior documentary bytes validated in preflight, no new fetch claim
    actions=pd.read_csv(R/'reports/p1_pit_v34/available_action_rows.csv');matches=[]
    for x in parsed:
        date=pd.to_datetime(x['ex_date'],format='%m/%d/%Y').date().isoformat();g=actions[(actions.asset==x['asset'])&(actions.ex_date==date)]
        assert len(g)==1 and abs(float(g.iloc[0].divCash)-x['distribution_per_share'])<=1e-10
        matches.append({**x,'normalized_ex_date':date,'issuer_payload_sha256':doc['sha256'] if doc else issuer['pdf_sha256'],'state':'VERIFIED_ACTION_PARTIAL','provider_pay_date_still_missing':True,'first_announcement_qualified':False,'full_coverage_qualified':False})
    delta={'fresh_payload_verified':bool(doc),'issuer_documents_equal_prior':bool(doc and doc['sha256']==issuer['pdf_sha256']),'before_matched':prior['matched_issuer_pay_dates'],'after_matched':len(matches),'new_unique_matches':len(matches)-prior['matched_issuer_pay_dates'],'provider_rows':len(actions),'provider_missing_pay_dates':int(actions.pay_date.isna().sum()),'unmatched_provider_actions':len(actions)-len(matches),'rows':matches,'P1_qualified':False,'provider_mutated':False};write('official_action_delta.json',delta);return delta

def inventory():
    # Aggregate inventory only. Do not retry the mixed-clock canonical bundle read.
    prior=load(R/'reports/p1_pit_v34/readiness.json');rows=[]
    for card in prior['sources']:
        for i in range(card['rows']):
            rows.append({'source':card['source'],'event_version':i,'entity':'NOT_REINSPECTED','field':'NOT_REINSPECTED','year':'NOT_REINSPECTED','canonical_file_sha256':card['sha256'],'before_tier_a':False,'after_tier_a':False,'state':'BLOCKED_SCOPE_BOUNDARY_REAUDIT','missing':'Need separately certified pre-2024-only canonical export; mixed clock-bound source audit stopped. Prior Tier-B aggregate inventory retained, NOT new per-version certification.'})
    frame=pd.DataFrame(rows);frame.to_csv(O/'evidence_qualification_matrix.csv',index=False)
    summary=[{'source':c['source'],'event_versions':c['rows'],'prior_tiers':c['tiers'],'after_tier_a':0,'per_version_reaudit_completed':False} for c in prior['sources']]
    out={'event_versions':len(frame),'before_qualified_tier_a':0,'after_qualified_tier_a':0,'per_source':frame.groupby('source').size().to_dict(),'all_row_identities_retained':True,'per_version_reaudit_completed':False,'aggregate_prior_receipt_sha256':file_hash(R/'reports/p1_pit_v34/readiness.json'),'per_source_status':summary,'blocker':'Combined canonical bundle failed 2024 clock boundary; no forbidden-row investigation or workaround performed. New exact source/entity/year recertification unavailable without a certified pre-2024 export.'};write('evidence_qualification_matrix.json',out);return out

def economics(delta):
    market=OT/'artifacts/tiingo_inputs/ba9be84158dedc267ba5191b298084ffa5cdc4e7dbe92bb105e828f7dd4062f0';validate_bundle(market);preparation=load(market/'p1_preparation.json');daily=pd.DataFrame(preparation['daily']);assert len(daily)==27088 and not daily.duplicated(['asset','session_date']).any();assert (pd.to_datetime(daily.session_date)<pd.Timestamp('2024-01-01')).all() and np.isfinite(daily.raw_open).all() and (daily.raw_open>0).all()
    actions=pd.DataFrame(preparation['actions']);matched={(x['asset'],x['normalized_ex_date']):x for x in delta['rows']};output=[];rh=file_hash(market/'p1_preparation.json')
    for x in daily.itertuples():
        a=actions[(actions.asset==x.asset)&(actions.ex_date==x.session_date)]
        # Audit instant only: not an invented trading decision/publication clock.
        s={'asset':x.asset,'session_date':x.session_date,'decision_at':x.session_date+'T00:00:00Z','raw_open':float(x.raw_open),'proofs':{},'action_status':'event_unqualified' if len(a) else 'unknown','test_not_evidence':False};q=p1_session(s);partial=(x.asset,x.session_date) in matched
        output.append({'asset':x.asset,'date':x.session_date,'year':int(x.session_date[:4]),'state':'VERIFIED_ACTION_PARTIAL' if partial else q['state'],'raw_open_bytes_verified':True,'raw_open_receipt_sha256':rh,'provider_actions':len(a),'issuer_pay_date_partial':matched[(x.asset,x.session_date)]['pay_date'] if partial else '', 'complete_action_or_no_action':False,'action_status':s['action_status'],'P1_eligible':q['P1_eligible'],'missing':'|'.join(q['missing'])})
    frame=pd.DataFrame(output);frame.to_csv(O/'p1_asset_date_qualification.csv',index=False);groups=frame.groupby(['asset','year']).agg(sessions=('date','size'),eligible=('P1_eligible','sum'),partial_issuer_events=('issuer_pay_date_partial',lambda x:int(x.ne('').sum()))).reset_index();groups['eligible_fraction']=groups.eligible/groups.sessions;groups.to_csv(O/'p1_asset_year_qualification.csv',index=False)
    out={'session_asset_rows':len(frame),'before_P1_qualified':0,'after_P1_qualified':int(frame.P1_eligible.sum()),'eligible_fraction':float(frame.P1_eligible.mean()),'asset_year_groups':len(groups),'issuer_pay_date_partial_rows':int(frame.issuer_pay_date_partial.ne('').sum()),'economic_PnL_computed':False,'cash_asof_verified':False,'action_absence_is_not_no_action':True,'per_asset':frame.groupby('asset').size().to_dict()};write('p1_coverage_summary.json',out);return out

def future():
    old=load(R/'p1_pit_readiness_v34/future_evaluation_plan_v1.json');v2=load(R/'configs/v33_minimal_execution_v2r1.json');bindings=[]
    for x in old['historical_candidate_bundles']:
        b=T/x['relative_bundle'];validate_bundle(b);assert file_hash(b/'model.pt')==x['model_sha256'];bindings.append(x)
    data=[]
    for name,relative in v2['data_bundles'].items():
        b=T/relative;validate_bundle(b);meta=load(b/'data.json');data.append({'track_year':name,'relative_bundle':relative,'data_receipt_hash':file_hash(b/'receipt.json'),'normalizer_id':meta['normalizer_id'],'target_id':meta['target_contract_id'],'comparison_grid_hash':digest(meta['keys'])})
    oof=pd.read_csv(R/'reports/calibration_v34/genuine_sia_oof.csv');thresholds={}
    for track,g in oof.groupby('track'):
        cutoff=pd.Timestamp('2020-01-01T00:00:00Z' if track=='Main-A' else '2022-01-01T00:00:00Z');assert (pd.to_datetime(g.label_available_at,utc=True)<cutoff).all();assert g.role.eq('OOF').all();v=((g.target-g.q50).abs()/g.scale).groupby(g.decision_time).mean();thresholds[track]={'definition':'q99 of equal-asset/seed-mean normalized absolute median OOF error per distinct date','quantile':.99,'threshold':float(v.quantile(.99)),'distinct_dates':len(v),'rows':len(g),'cutoff':cutoff.isoformat(),'last_label_available':pd.to_datetime(g.label_available_at,utc=True).max().isoformat(),'state':'TRAIN_ONLY_CANDIDATE_NOT_OPERATIONALLY_FROZEN','source_hash':file_hash(R/'reports/calibration_v34/genuine_sia_oof.csv'),'outer_labels_used':False}
    write('mature_OOF_monitor_candidates.json',thresholds)
    plan={'schema_version':'v35-future-prepared-1','historical_research_candidate':'SIA v2 only; no scientific promotion or refit','candidate_models':bindings,'data_contracts':data,'threshold_candidates':thresholds,'baseline_candidates':{'Main-A':'historical I0 matched fold reference','Nested-B':'linear_quantile I0 matched fold reference'},'baseline_frozen_receipt':{'path':'reports/v33_minimal_results_v2.json','sha256':file_hash(R/'reports/v33_minimal_results_v2.json')},'seeds':[11,37,71],'assets':['SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'],'quantiles':[.05,.1,.5,.9,.95],'metric':'date_equal_asset_equal_seed_average_train_scale_normalized_pinball','uncertainty':{'bootstrap_draws':2000,'blocks':[4,8,13],'seeds_independent_market_units':False},'cohorts':{'per_track':2,'distinct_dates_each':52,'nonoverlapping':True,'mature_labels_only':True},'signed_freeze_required':True,'independent_reviewer_required':True,'exact_future_benchmark_and_model_binding_required':True,'actual_forecast_issuance_in_this_task':False,'retrospective_reserved_access':False,'scheduled_dates':None,'source_versions_all_tier_a':False,'P1_qualified':False,'no_outer_retuning':True};plan['bundle_id']=sha(canonical(plan));write('future_prepared_bundle.json',plan)
    flags={'candidate_hashes_verified':True,'benchmark_hashes_verified':True,'normalizers_verified':True,'all_source_versions_tier_a':False,'all_P1_sessions_qualified':False,'independent_reviewer':False,'thresholds_mature_train_only':True,'operational_freeze_verified':False,'scientific_freeze_verified':False,'user_candidate_selection':False};out=readiness(flags);out.update(flags=flags,readiness_bundle_id=plan['bundle_id'],state='BLOCKED_EVIDENCE',historical_benchmark_binding_only=True,future_exact_benchmark_binding_qualified=False,actual_mature_future_dates=0,implementation_ready=True);write('freeze_readiness.json',out);return out

def run():
    plan=load(Q/'pilot_scope_v1.json');assert plan['plan_id']==digest({k:v for k,v in plan.items() if k!='plan_id'});emit('PHASE_B_ACQUISITION',plan_id=plan['plan_id']);records=acquire(plan);delta=issuer_delta(records);emit('PHASE_B_CANONICAL_INVENTORY');inv=inventory();emit('PHASE_C_P1_QUALIFICATION');p1=economics(delta);emit('PHASE_D_FREEZE_READY_VALIDATOR');f=future()
    write('execution.json',{'created_at':now(),'code_hash':hashcode(),'pilot_plan_id':plan['plan_id'],'inventory':inv,'P1':p1,'issuer_matches':len(delta['rows']),'new_matches':delta['new_unique_matches'],'freeze_state':f['state'],'GPU_training':0,'reserved_access':False,'original_source_hash':code_hash(ORIG),'frozen_v2_hash':code_hash(R)});emit('REAL_EVIDENCE_PHASES_EXECUTED',tier_a=inv['after_qualified_tier_a'],P1=p1['after_P1_qualified'],issuer_partial=24,forward='BLOCKED_EVIDENCE')
if __name__=='__main__':run()
