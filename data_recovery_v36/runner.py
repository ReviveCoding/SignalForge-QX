"""V36 documentary parsing/P1 gap/shadow fixture execution. No model/outer targets."""
import io,json,csv,re,zipfile,subprocess,importlib.util
from pathlib import Path
from datetime import datetime,timedelta,timezone
from html.parser import HTMLParser
import pandas as pd
from signalforge.runtime import file_hash,code_hash,validate_bundle,digest,now
from qualification_v35.core import canonical,sha,EVIDENCE_FIELDS,qualify,p1_session
from data_recovery_v36.acquire import R,T,Q,O,load,append_progress
from data_recovery_v36.core import observation,cash_rows,action_join,historical_date,documentary_tier
from data_recovery_v36.shadow import ShadowRecorder,replay,validate_message
OT=Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1')
FIELDS={'Commercial (Excluding SPR)':'commercial_crude','Cushing':'cushing','SPR':'SPR','Total Motor Gasoline':'gasoline','Distillate Fuel Oil':'distillate'}
def write(name,value):
    with (O/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def frame(name,rows):pd.DataFrame(rows).to_csv(O/name,index=False)
def pdf_text(path):
    code='import fitz,sys,json; d=fitz.open(sys.argv[1]); print(json.dumps([{ "page":i+1,"text":p.get_text()} for i,p in enumerate(d)]))'
    return json.loads(subprocess.check_output([str(OT/'tools/pdf_qa_env/bin/python'),'-c',code,str(path)],text=True,timeout=30))
def stocks_csv(payload,h,version):
    raw=list(csv.reader(io.StringIO(payload.decode('utf-8-sig'))));current=datetime.strptime(raw[0][1],'%m/%d/%y').date().isoformat();prior=datetime.strptime(raw[0][2],'%m/%d/%y').date().isoformat();out=[]
    for row in raw[1:]:
        if row[0] not in FIELDS:continue
        a,b,c=[float(x.replace(',','')) for x in row[1:4]]
        out.append({**observation('EIA','US',FIELDS[row[0]],current,a,'million_barrels',h,version),'prior_reference_date':prior,'prior_value':b,'reported_change':c,'arithmetic_error':a-b-c,'arithmetic_tolerance':.0021,'arithmetic_pass':abs(a-b-c)<=.0021})
    if len(out)!=5:raise ValueError('Five registered stock fields required')
    return out
class Plain(HTMLParser):
    def __init__(self):super().__init__();self.words=[]
    def handle_data(self,x):self.words.append(x)
def parse_documents():
    acquired=load(O/'attempted_public_sources.json');inventory=[];observations=[];eia=[];cash=[];cftc=[];claims=[];qualifications=[];failures=[];issuer=[]
    for d in acquired['records']:
        if not d.get('payload_path'):
            inventory.append({**{k:d.get(k) for k in ['family','url','state','status']},'parsed':False,'tier':'BLOCKED_NO_PAYLOAD','first_public_qualified':False});continue
        p=Path(d['payload_path']);b=p.read_bytes();assert sha(b)==d['sha256'];url=d['url'];family=d['family'];text='';extracted=[];status='DOCUMENT_CONTENT_ONLY';rows=0
        try:
            if '.pdf' in url:
                pages=pdf_text(p);text='\n'.join(x['text'] for x in pages);extracted=pages
                if 'distribution-summary' in url:
                    for page in pages:
                        lines=page['text'].splitlines()
                        for i,line in enumerate(lines):
                            if line.strip() not in ['IEF','TLT']:continue
                            seg=lines[i:i+12];dates=[x.strip() for x in seg if re.fullmatch(r'\d\d/\d\d/2023',x.strip())]
                            if len(dates)!=3:raise ValueError('Issuer three-date schema')
                            amount=next(x.strip() for x in seg[seg.index(dates[-1])+1:] if re.fullmatch(r'\d+\.\d+',x.strip()))
                            issuer.append({'asset':line.strip(),'record_date':datetime.strptime(dates[0],'%m/%d/%Y').date().isoformat(),'ex_date':datetime.strptime(dates[1],'%m/%d/%Y').date().isoformat(),'pay_date':datetime.strptime(dates[2],'%m/%d/%Y').date().isoformat(),'amount':float(amount),'pdf_page':page['page'],'payload_sha256':d['sha256']})
                    rows=len(issuer);status='PARTIAL_ACTUAL_ISSUER_EX_PAY_AMOUNT'
                elif 'table4.pdf' in url:
                    if 'Million Barrels' not in text:raise ValueError('EIA unit header')
                    lines=[x.strip() for x in text.splitlines()];release=re.search(r'/2023/(2023_\d\d_\d\d)/',url).group(1)
                    for label,field in FIELDS.items():
                        indexes=[i for i,x in enumerate(lines) if x.startswith(label)]
                        if len(indexes)!=1:raise ValueError('EIA PDF exact field unique')
                        i=indexes[0];vals=[]
                        for x in lines[i+1:i+9]:
                            if re.fullmatch(r'-?\d[\d,]*\.\d+',x):vals.append(float(x.replace(',','')))
                            else:break
                        if len(vals)!=7:raise ValueError('EIA PDF seven numeric columns')
                        eia.append({'format':'pdf','release':release,'field':field,'value':vals[0],'prior_value':vals[1],'reported_change':vals[2],'arithmetic_error':vals[0]-vals[1]-vals[2],'arithmetic_tolerance':.1501,'arithmetic_pass':abs(vals[0]-vals[1]-vals[2])<=.1501,'payload_sha256':d['sha256'],'unit':'million_barrels'})
                    rows=5;status='STOCK_LEVELS_EXTRACTED_NOT_CLOCK_QUALIFIED'
            elif 'table4.csv' in url:
                release=re.search(r'/2023/(2023_\d\d_\d\d)/',url).group(1);records=stocks_csv(b,d['sha256'],release);observations+=records
                eia +=[{**x,'format':'csv','release':release} for x in records];rows=len(records);text=b.decode();status='STOCK_LEVELS_EXTRACTED_NOT_CLOCK_QUALIFIED'
            elif family=='CASH_NYFED':
                records=cash_rows(b,d['sha256']);assert all(x['reference_date'].startswith('2023-') for x in records);cash+=records;observations+=records;rows=len(records);text=b.decode();status='SOURCE_SEMANTICS_VERIFIED_NOT_ASOF_CASH';extracted=records
            elif family=='CFTC':
                z=zipfile.ZipFile(io.BytesIO(b));infos=z.infolist();assert len(infos)==1 and infos[0].file_size<32*1024*1024 and not infos[0].is_dir();table=pd.read_csv(io.BytesIO(z.read(infos[0])));dates=pd.to_datetime(table.Report_Date_as_YYYY_MM_DD if 'Report_Date_as_YYYY_MM_DD' in table else table['Report_Date_as_YYYY-MM-DD']);assert (dates.dt.year==2023).all();keys=['CFTC_Contract_Market_Code','Report_Date_as_YYYY-MM-DD'];assert not table.duplicated(keys).any();oi=pd.to_numeric(table.Open_Interest_All);assert oi.ge(0).all();rows=len(table)
                records=[observation('CFTC',str(row.CFTC_Contract_Market_Code),'open_interest',date.date().isoformat(),float(row.Open_Interest_All),'contracts',d['sha256'],d['sha256']) for row,date in zip(table.itertuples(),dates)]
                observations+=records;cftc.append({'url':url,'raw_rows':rows,'distinct_report_dates':int(dates.nunique()),'distinct_markets':int(table.CFTC_Contract_Market_Code.nunique()),'minimum_date':dates.min().date().isoformat(),'maximum_date':dates.max().date().isoformat(),'positioning_not_flow':True,'correction_lineage_complete':False,'first_public_qualified':False});extracted=records;status='ANNUAL_ARCHIVE_POSITIONING_NOT_ORIGINAL_WEEKLY_CLOCK'
            elif url.endswith('.xls'):
                if not importlib.util.find_spec('xlrd'):status='BLOCKED_XLS_PARSER_DEPENDENCY';failures.append({'url':url,'error_type':'MissingOptionalXlrd','no_global_install':True})
                else:raise RuntimeError('XLS parser not preregistered')
            else:
                html=b.decode(errors='replace');parser=Plain();parser.feed(html);text=' '.join(parser.words);extracted=[{'text':text}]
                if family=='FED_G17':
                    # Documentary release table, not ALFRED-vintage-to-canonical match.
                    tables=pd.read_html(io.StringIO(html));summaries=[]
                    for i,table in enumerate(tables):
                        summaries.append({'table':i,'rows':len(table),'columns':[str(x) for x in table.columns]})
                    rows=sum(x['rows'] for x in summaries);extracted={'tables':summaries,'text':text};status='G17_ARCHIVED_TABLES_UNIT_BASE_NOT_CANONICAL_LEVEL_RELABEL'
            if extracted:
                outfile=O/('extracted_'+str(d['candidate_number'])+'.json');outfile.write_text(json.dumps(extracted,indent=2,allow_nan=False));extraction=file_hash(outfile)
            else:extraction=sha(text.encode())
        except Exception as error:
            status='PARSER_FAILED_PRESERVED';failures.append({'url':url,'error_type':type(error).__name__,'message':str(error),'payload_sha256':d['sha256']});extraction=None
        tier=documentary_tier(d,text);inventory.append({**{k:d.get(k) for k in ['family','url','state','status','sha256','bytes','retrieved_at']},'payload_path':str(p),'parsed':status not in ['PARSER_FAILED_PRESERVED','BLOCKED_XLS_PARSER_DEPENDENCY'],'parser_state':status,'extracted_sha256':extraction,'extracted_rows':int(rows),'tier':tier['tier'],'first_public_qualified':False})
        e={k:None for k in EVIDENCE_FIELDS};e.update(schema_version='v35-evidence-1',claim_id='v36-document-'+str(d['candidate_number']),source=family,publisher=url.split('/')[2],url=url,entity='DOCUMENT_ONLY',field='retrieved_historical_publisher_content',version_id=d['sha256'],payload_sha256=d['sha256'],parser_sha256=file_hash(Q/'runner.py'),valid_at=d['retrieved_at'],known_at=d['retrieved_at'],first_public_at=None,retrieved_at=d['retrieved_at'],clock_basis='retrieval',version_kind='original',correction_chain=[],correction_history_complete=False,scope=['independently scoped pre2024 historical document, not old canonical event export'],scope_complete=False,test_not_evidence=False);claims.append(e);qualifications.append(qualify(e,b,None))
    frame('verified_originals.csv',inventory);frame('public_observation_versions.csv',observations);frame('eia_stock_documentary_values.csv',eia);frame('cash_documentary_rates.csv',cash);write('CFTC_archive_summary.json',cftc);write('evidence_inbox.json',{'claims':claims,'trusted_publisher_keys':{},'trusted_reviewers':{}});write('document_qualification.json',{'results':qualifications,'Tier_A':sum(x['state']=='QUALIFIED_TIER_A' for x in qualifications)});write('parser_incidents.json',failures)
    comparisons=[];ef=pd.DataFrame(eia)
    for release,g in ef.groupby('release'):
        a=g[g.format=='csv'].set_index('field');b=g[g.format=='pdf'].set_index('field')
        for field in a.index.intersection(b.index):
            comparisons.append({'release':release,'field':field,'comparison':'CSV_VS_PDF_SAME_RELEASE','reference_date':a.loc[field,'reference_date'],'CSV':a.loc[field,'value'],'PDF':b.loc[field,'value'],'difference':a.loc[field,'value']-b.loc[field,'value'],'rounding_tolerance':.0501,'within_rounding':bool(abs(a.loc[field,'value']-b.loc[field,'value'])<=.0501),'CSV_arithmetic_pass':bool(a.loc[field,'arithmetic_pass']),'PDF_arithmetic_pass':bool(b.loc[field,'arithmetic_pass']),'payloads':[a.loc[field,'payload_sha256'],b.loc[field,'payload_sha256']]})
    csvrows=ef[ef.format=='csv']
    for _,new in csvrows.iterrows():
        old=csvrows[(csvrows.reference_date==new.prior_reference_date)&(csvrows.field==new.field)]
        for _,old in old.iterrows():comparisons.append({'release':new.release,'field':new.field,'comparison':'LATER_RELEASE_PRIOR_VS_EARLIER_CURRENT','reference_date':old.reference_date,'earlier_current':old.value,'later_prior':new.prior_value,'difference':new.prior_value-old.value,'state':'OBSERVED_CROSS_RELEASE_LEVEL_CHANGE' if abs(new.prior_value-old.value)>.0021 else 'MATCH_WITHIN_ROUNDING','payloads':[old.payload_sha256,new.payload_sha256],'original_revision_lineage_proven':False})
    write('EIA_documentary_discrepancies.json',{'comparisons':comparisons,'arithmetic_failures':[{k:v for k,v in x.items() if k in ['release','field','format','value','prior_value','reported_change','arithmetic_error','arithmetic_tolerance','payload_sha256']} for x in eia if not x['arithmetic_pass']],'canonical_comparison':'BLOCKED_NO_CERTIFIED_PRE2024_EXPORT','appendix_propane_not_Table4_patch':True,'canonical_mutated':False})
    actions=pd.read_csv(R/'reports/p1_pit_v34/available_action_rows.csv').fillna('');provider=actions.to_dict('records');matches=[]
    for x in issuer:matches.append({**x,**action_join(x,provider),'provider_pay_date_missing':True,'new_unique_vs_v35':False})
    assert len(matches)==24 and all(x['matches']==1 for x in matches);frame('issuer_actions_delta.csv',matches)
    write('public_analysis.json',{'documents':len(claims),'parsed_documents':sum(x.get('parsed',False) for x in inventory),'observation_versions':len(observations),'cash_rates':len(cash),'issuer_partial_matches':len(matches),'new_unique_issuer_matches':0,'cftc_rows':sum(x['raw_rows'] for x in cftc),'Tier_A':0,'P1':0,'parser_incidents':len(failures),'original_first_dissemination_proofs':0,'corrected_original_payload_pairs':0,'split_completeness_certificates':0})
    return cash,matches

def p1_gaps(cash,matches):
    f=pd.read_csv(R/'reports/qualification_v35/p1_asset_date_qualification.csv');assert len(f)==27088 and not f.P1_eligible.any() and (f.date<'2024-01-01').all()
    cash_dates={x['reference_date'] for x in cash};f['cash_documentary_date_present']=f.date.isin(cash_dates);f['asof_cash_qualified']=False;f['first_public_qualified']=False;f.to_csv(O/'p1_scope_gaps.csv',index=False)
    g=f.groupby(['asset','year']).agg(sessions=('date','size'),partial_issuer_events=('issuer_pay_date_partial',lambda x:int(x.notna().sum())),documentary_cash_dates=('cash_documentary_date_present','sum'),qualified=('P1_eligible','sum')).reset_index();g.to_csv(O/'p1_asset_year_gaps.csv',index=False)
    inv=load(R/'reports/qualification_v35/evidence_qualification_matrix.json');frame('source_version_readiness.csv',[{'source':source,'prior_aggregate_versions':count,'per_version_state':'BLOCKED_SCOPE_BOUNDARY_NOT_REAUDITED','Tier_A_before':0,'Tier_A_after':0,'new_linked_original_versions':0,'need':'Independently certified pre2024-only source/version export'} for source,count in inv['per_source'].items()]);write('genuine_certification_receipt.json',{'before_Tier_A':0,'after_Tier_A':0,'prior_aggregate_versions':inv['event_versions'],'new_document_claims':load(O/'public_analysis.json')['documents'],'canonical_version_links':0,'new_P1_sessions':0,'P1_denominator':len(f),'raw_open_bytes_available':len(f),'raw_execution_open_or_publication_qualified':False,'issuer_partial_unique_before':24,'issuer_partial_unique_after':len(matches),'new_unique_issuer_matches':0,'provider_missing_pay_dates':438,'issuer_unmatched_provider_actions':414,'asset_years':len(g),'cash_documentary_asset_sessions':int(f.cash_documentary_date_present.sum()),'cash_asof_qualified_sessions':0,'new_scientific_signers':0,'economic_PnL_computed':False})
    bottlenecks=[{'dependency':'Scoped canonical export + perversion original/correction/dissemination proof','potential_scope':'67448 prior aggregate versions','availability':'external custodian certification/logs needed','remaining_joint_gates':'trusted publisher keys, source/field/unit join, independent review','guaranteed_unlock_now':0},{'dependency':'Raw tradable opens and publication/permission proof','potential_scope':'27088 asset sessions','availability':'raw price bytes exist; attestation and entitlement missing','remaining_joint_gates':'actions/cash/cost accounting/review','guaranteed_unlock_now':0},{'dependency':'Complete action + signed no-action + split histories','potential_scope':'8 assets x 14 years (112 groups); 438 known provider action rows','availability':'24 issuer pay tuples; 414 other action rows and no-action completeness missing','remaining_joint_gates':'first-known/pay/split/price/cash/review','guaranteed_unlock_now':0},{'dependency':'As-of cash benchmark instrument and timing','potential_scope':str(int(f.cash_documentary_date_present.sum()))+' asset sessions have 2023 official rate quotes','availability':'NYFed official rates exist; publication-asof and executable accrual conventions absent','remaining_joint_gates':'all other P1 gates','guaranteed_unlock_now':0},{'dependency':'Independent reviewer + operational/scientific freeze + candidate/control selection','potential_scope':'Future P0 lane; does not require P1 economics','availability':'genuine external authority and selection absent','remaining_joint_gates':'all TierA sources, exact forward benchmark, future observations','guaranteed_unlock_now':0}];frame('evidence_bottleneck_dashboard.csv',bottlenecks)

def shadow_fixture():
    old=load(R/'reports/qualification_v35/future_prepared_bundle.json');readiness=load(R/'reports/qualification_v35/freeze_readiness_v2.json')
    plan={'schema':'v36-shadow-test-ready-1','parent_prepared_bundle_hash':file_hash(R/'reports/qualification_v35/future_prepared_bundle.json'),'model_hashs':[x['model_sha256'] for x in old['candidate_models']],'source_hashs':[x['data_receipt_hash'] for x in old['data_contracts']],'normalizer_hashs':[x['normalizer_id'] for x in old['data_contracts']],'target_hashs':[x['target_id'] for x in old['data_contracts']],'benchmark_hashs':[old['baseline_frozen_receipt']['sha256']],'assets':old['assets'],'seeds':old['seeds'],'quantiles':old['quantiles'],'cohorts':old['cohorts'],'uncertainty':old['uncertainty'],'scheduled_dates':'UNKNOWN','real_issuance_allowed':False,'historical_bindings_only':True,'missing_P0_gates':readiness['prediction_missing'],'P1_separate':True,'no_daemon':True,'real_source_or_current_market_access':False};plan['plan_id']=digest(plan)
    write('shadow_prepared_plan.json',plan);db=T/'ledger/data_recovery_v36/shadow_test_only.sqlite';rec=ShadowRecorder(db,plan);base={'kind':'TEST_ONLY','retry_key':'v36-dry-1','track':'Main-A','asset':'SPY','seed':11,'decision_at':'2021-01-08T00:00:00Z','source_valid_at':'2021-01-06T00:00:00Z','source_known_at':'2021-01-07T00:00:00Z','expiry_at':'2021-01-09T00:00:00Z','label_end':'2021-01-15T00:00:00Z','label_available_at':'2021-01-16T00:00:00Z','block':0,'quantiles':[-2,-1,0,1,2]}
    for k in ['model_hash','source_hash','normalizer_hash','target_hash','benchmark_hash']:base[k]=plan[k+'s'][0]
    receipts=[]
    for seed in plan['seeds']:
        for asset in plan['assets']:
            m={**base,'seed':seed,'asset':asset,'retry_key':'v36-dry-'+asset+'-'+str(seed)};receipts.append(rec.append(m))
    retry=rec.append({**base,'retry_key':'v36-dry-SPY-11'});assert retry['reused'];conflict=False
    try:rec.append({**base,'kind':'REAL_FORWARD'})
    except PermissionError:conflict=True
    assert conflict;write('shadow_fixture_input.json',{**base,'retry_key':'v36-dry-SPY-11'});result=replay(db);write('shadow_readiness_receipt.json',{'database':str(db),'plan_id':plan['plan_id'],'replay':result,'dry_run_appends':len(receipts),'actual_clock_observation':now(),'actual_forecasts':0,'real_request_blocked':conflict,'duplicate_retry_reused':retry['reused'],'P0':readiness['prediction_state'],'P1':readiness['economics_state'],'future_mature_dates':0,'scheduled_dates':'UNKNOWN','TEST_ONLY':True});return result

def main():
    append_progress('DOCUMENT_PARSING_START');cash,matches=parse_documents();append_progress('DOCUMENT_PARSING_PASS',**load(O/'public_analysis.json'));p1_gaps(cash,matches);append_progress('P1_GAPS_COMPLETE',qualified=0,denominator=27088);s=shadow_fixture();append_progress('SHADOW_TEST_ONLY_PASS',records=s['records'],real_forecasts=0)
if __name__=='__main__':main()
