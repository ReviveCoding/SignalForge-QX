"""Independent v35 scalar counts/provenance/preservation checks. No GPU or HTTP."""
import os,sys,json,subprocess
from pathlib import Path
import pandas as pd,numpy as np
from signalforge.runtime import file_hash,code_hash,validate_bundle,digest,now,commit_bundle
from qualification_v35.core import canonical,sha,qualify,EVIDENCE_FIELDS,P1_REQUIREMENTS
from qualification_v35.runner import R,T,OT,ORIG,O,Q,load,write,hashcode
from diagnostics_v3.audit import preserve

def supplemental_documents():
    records=load(O/'verified_publisher_documents.json')['records'];claims=[];results=[];semantics=[]
    for i,r in enumerate(records):
        if not r.get('relative_bundle'):continue
        b=T/r['relative_bundle'];validate_bundle(b);payload=(b/r['filename']).read_bytes();assert sha(payload)==r['sha256']
        e={k:None for k in EVIDENCE_FIELDS};e.update(schema_version='v35-evidence-1',claim_id='official-document-'+str(i),source='official_public_document',publisher=r['url'].split('/')[2],url=r['url'],entity='DOCUMENT_ONLY',field='document_content_as_retrieved_NOT_historical_event_clock',version_id=r['sha256'],payload_sha256=r['sha256'],parser_sha256=file_hash(Q/'core.py'),valid_at=r['retrieved_at'],known_at=r['retrieved_at'],first_public_at=None,retrieved_at=r['retrieved_at'],clock_basis='retrieval',version_kind='original',correction_chain=[],correction_history_complete=False,scope=['issuer2023/EIA2023/docsemantics; not canonical event qualification'],scope_complete=False,test_not_evidence=False)
        value=qualify(e,payload,None);assert value['state']=='VERIFIED_PAYLOAD_BUT_NO_FIRST_PUBLIC';claims.append(e);results.append(value)
        if r['filename'].endswith('.pdf'):
            code='import fitz,sys; d=fitz.open(sys.argv[1]); print("\\n".join(p.get_text() for p in d))'
            text=subprocess.check_output([str(OT/'tools/pdf_qa_env/bin/python'),'-c',code,str(b/r['filename'])],text=True,timeout=30)
            semantics.append({'url':r['url'],'payload_sha256':r['sha256'],'extracted_text_sha256':sha(text.encode()),'document_scope':'2023 issuer distributions retrospective tax supplement' if 'ishares' in r['url'] else '2023-08-30 original archive Table4 as presently downloaded' if 'table4' in r['url'] else 'propane/propylene correction appendix; not revised crude/gasoline/distillate Table4','mentions_propane':'propane' in text.lower(),'mentions_revision':'revis' in text.lower(),'first_dissemination_proven':False})
    write('evidence_inbox_v1.json',{'schema_version':'v35-evidence-1','records':claims,'no_trusted_keys_enrolled':True,'document_retrieval_fields_are_not_historical_event_availability':True});write('document_qualification.json',{'claims':results,'actual_tier_a_promotions':0,'document_semantics':semantics})
    schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'V35 externally authenticated event-version evidence','type':'object','additionalProperties':False,'required':sorted(EVIDENCE_FIELDS),'properties':{k:{} for k in sorted(EVIDENCE_FIELDS)}}
    for k in ['schema_version','claim_id','source','publisher','url','entity','field','version_id','payload_sha256','parser_sha256','valid_at','known_at','retrieved_at','clock_basis','version_kind']:schema['properties'][k]={'type':'string'}
    for k in ['payload_sha256','parser_sha256']:schema['properties'][k]['pattern']='^[0-9a-f]{64}$'
    for k in ['scope_complete','correction_history_complete','test_not_evidence']:schema['properties'][k]={'type':'boolean'}
    for k in ['scope','correction_chain']:schema['properties'][k]={'type':'array'}
    schema['properties']['schema_version']={'const':'v35-evidence-1'};schema['properties']['first_public_at']={'type':['string','null']};schema['properties']['dissemination_record']={'type':['object','null']};schema['properties']['reviewer_record']={'type':['object','null']}
    if (Q/'evidence.schema.json').exists():assert load(Q/'evidence.schema.json')==schema
    else:
        with (Q/'evidence.schema.json').open('x') as f:json.dump(schema,f,indent=2)


def independent():
    supplemental_documents();pre=load(O/'preflight.json');plan=load(Q/'pilot_scope_v1.json');assert plan['plan_id']==digest({k:v for k,v in plan.items() if k!='plan_id'})
    docs=load(O/'verified_publisher_documents.json')['records'];assert len(docs)==5 and all(x['attempt']==1 for x in docs);success=[x for x in docs if x.get('relative_bundle')];assert len(success)==4
    for x in success:validate_bundle(T/x['relative_bundle']);assert file_hash(T/x['relative_bundle']/x['filename'])==x['sha256']
    delta=load(O/'official_action_delta.json');assert delta['before_matched']==delta['after_matched']==24 and delta['new_unique_matches']==0
    actions=pd.read_csv(R/'reports/p1_pit_v34/available_action_rows.csv');assert len(actions)==438 and actions.pay_date.isna().sum()==438
    for x in delta['rows']:
        g=actions[(actions.asset==x['asset'])&(actions.ex_date==x['normalized_ex_date'])];assert len(g)==1;assert abs(float(g.iloc[0].divCash)-x['distribution_per_share'])<1e-10
    p1=pd.read_csv(O/'p1_asset_date_qualification.csv');assert len(p1)==27088 and not p1.duplicated(['asset','date']).any();assert not p1.P1_eligible.any();assert (pd.to_datetime(p1.date)<pd.Timestamp('2024-01-01')).all();assert p1.issuer_pay_date_partial.notna().sum()==24
    market=OT/'artifacts/tiingo_inputs/ba9be84158dedc267ba5191b298084ffa5cdc4e7dbe92bb105e828f7dd4062f0';validate_bundle(market);daily=pd.DataFrame(load(market/'p1_preparation.json')['daily']);assert set(zip(daily.asset,daily.session_date))==set(zip(p1.asset,p1.date));assert np.isfinite(daily.raw_open).all() and (daily.raw_open>0).all()
    grouping=p1.groupby(['asset','year']).size();reported=pd.read_csv(O/'p1_asset_year_qualification.csv').set_index(['asset','year']);assert grouping.to_dict()==reported.sessions.to_dict();assert len(grouping)==112
    versions=pd.read_csv(O/'evidence_qualification_matrix.csv');assert len(versions)==67448 and not versions.after_tier_a.any();assert versions.state.eq('BLOCKED_SCOPE_BOUNDARY_REAUDIT').all();assert not load(O/'evidence_qualification_matrix.json')['per_version_reaudit_completed']
    future=load(O/'future_prepared_bundle.json');assert future['bundle_id']==sha(canonical({k:v for k,v in future.items() if k!='bundle_id'}));assert future['scheduled_dates'] is None and not future['actual_forecast_issuance_in_this_task']
    for m in future['candidate_models']:validate_bundle(T/m['relative_bundle']);assert file_hash(T/m['relative_bundle']/'model.pt')==m['model_sha256']
    frame=pd.read_csv(R/'reports/calibration_v34/genuine_sia_oof.csv');thresholds=load(O/'mature_OOF_monitor_candidates.json')
    for track,g in frame.groupby('track'):
        err=(g.target-g.q50).abs()/g.scale;dates=pd.DataFrame({'date':g.decision_time,'err':err}).groupby('date').err.mean();expected=float(np.quantile(dates.values,.99));assert abs(expected-thresholds[track]['threshold'])<1e-12;assert len(dates)==thresholds[track]['distinct_dates'];assert (pd.to_datetime(g.label_available_at,utc=True)<pd.Timestamp(thresholds[track]['cutoff'])).all()
    # Scoped handoff tests, no old historical test receipts rewritten.
    env=dict(os.environ,PYTHONPATH=':'.join(str(p) for p in [R,R/'src',R/'scripts',R/'studies_v3']));p=subprocess.run([sys.executable,'-m','pytest','qualification_v35/test_contract.py','calibration_v34/test_contract.py','calibration_v34/serialization_r1/test_adapter.py','p1_pit_readiness_v34/test_gates.py','p1_pit_readiness_v34/test_bundle_names.py','tests/v33/test_budget_free_policy.py','--import-mode=importlib','-q','-p','no:cacheprovider'],capture_output=True,text=True,env=env);print(p.stdout+p.stderr,flush=True);write('independent_tests_r1.json',{'exit_code':p.returncode,'output':p.stdout+p.stderr,'historical_failures_not_modified':True});assert p.returncode==0
    for path,h in pre['protected_files'].items():assert file_hash(R/path)==h,path
    preservation=preserve();idx=load(R/'reports/calibration_v34/final_index.json')
    for path,h in idx['files'].items():assert file_hash(R/path)==h,path
    validate_bundle(T/idx['relative_bundle']);fitrows=load(R/'reports/calibration_v34/oof_completion.json')['results']
    for fit in fitrows:validate_bundle(T/fit['relative_bundle']);assert file_hash(fit['checkpoint_path'])==fit['checkpoint_hash']
    for path,h in load(R/'reports/model_risk_v3/final_handoff.json')['files'].items():assert file_hash(R/path)==h,path
    result={'passed':True,'created_at':now(),'code_hash':hashcode(),'v2_source_hash':code_hash(R),'original_source_hash':code_hash(ORIG),'protected_files_verified':len(pre['protected_files']),'v34_models_checkpoints_verified':len(fitrows),'v34_report_files_verified':len(idx['files']),'official_request_attempts':5,'verified_document_payloads':4,'http_403_retained_no_retry':1,'actual_tier_a_promotions':0,'P1_qualified_sessions':0,'P1_raw_open_sessions':27088,'P1_asset_years':112,'issuer_matches':24,'new_issuer_matches':0,'prior_canonical_version_inventory':67448,'new_per_version_canonical_reaudit':'BLOCKED_SCOPE_BOUNDARY','independent_threshold_counts':{k:v['distinct_dates'] for k,v in thresholds.items()},'preservation':preservation,'tests':{'exit_code':p.returncode,'output':p.stdout+p.stderr},'future_observations':0,'forecasts_issued':0,'GPU_training':0,'reserved_or_final_outcomes_read':False,'incident':'Initial boundary audit aborted; forbidden-row follow-up rejected and not executed; prior aggregate counts only used afterwards.'};write('independent_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['preservation','tests']}),flush=True)
if __name__=='__main__':independent()
