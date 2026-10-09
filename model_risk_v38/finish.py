"""Seal only after fresh whole-v38 independent output/preservation verification."""
from .source import *
from .independent import dt,linear_quantile
from signalforge.runtime import validate_bundle,commit_bundle,digest,code_hash
import math

def validate_final():
 prior=readj(OUT/'independent_validation.json');tests=readj(OUT/'tests_final.json')
 if not prior['passed'] or tests['exit_code'] or any(tests['counts'][k] for k in ['failures','errors','skipped']):raise ValueError('Validation scope failed')
 writej('test_count_reconciliation.json',{'new_test_cases':75,'prior_pure_cases':23,'total':98,'canonical_evidence':'tests_final.json.per_file_test_cases','note':'Initial runner descriptive text assumed24 prior; actual collected cases are23, all passed; original receipt retained'})
 protection=preservation()
 if protection!=readj(OUT/'preservation_start.json'):raise ValueError('Preservation mismatch')
 if code_hash(ORIGINAL)!='7cdf374c83afd29e07c79e2ea8a0cc10ff2cad3637cee6ff01fa95ba9f7049bf':raise ValueError('Original source hash mismatch')
 for rel,h in readj(OUT/'source_index.json')['inputs'].items():
  if sha(R/rel)!=h:raise ValueError('Changed source panel')
 assert sha(OUT/'frozen_plan.json')==readj(OUT/'source_index.json')['plan_sha256']
 # All lineage verified again, without loading/training any model.
 c=readj(R/'reports/calibration_v34/oof_completion.json');fitcount=0
 for f in c['results']:
  validate_bundle(T/f['relative_bundle'])
  if sha(f['checkpoint_path'])!=f['checkpoint_hash']:raise ValueError('Checkpoint drift')
  fitcount+=1
 idx=readj(R/'reports/calibration_v34/final_index.json')
 for rel,h in idx['files'].items():
  if sha(R/rel)!=h:raise ValueError('v34 report drift')
 validate_bundle(T/idx['relative_bundle']);handoff=readj(R/'reports/evidence_procurement_v37/handoff.json');v37=validate_bundle(T/handoff['relative_bundle'])
 data=pd.read_csv(R/'reports/calibration_v34/outer_forecasts.csv');data.decision_time=aware(data.decision_time);table=data.set_index(['track','model','decision_time','asset','seed']).sort_index();cur=pd.read_csv(OUT/'challenger_row_disagreement.csv');cur.decision_time=aware(cur.decision_time)
 q=np.array([.05,.1,.5,.9,.95]);maxdelta=0.
 for row in cur.itertuples():
  a=table.loc[(row.track,row.candidate,row.decision_time,row.asset,row.seed)];b=table.loc[(row.track,row.control,row.decision_time,row.asset,row.seed)];loss=[]
  for v in [a,b]:
   e=v.target-v[QC].to_numpy(float);loss.append(float(np.maximum(q*e,(q-1)*e).mean()/v.scale))
  maxdelta=max(maxdelta,abs((loss[0]-loss[1])-row.delta))
  if (loss[0]<loss[1])!=row.corrects_loss or (loss[0]>loss[1])!=row.hurts_loss:raise ValueError('Wrong correction/hurt attribution')
 if maxdelta>1e-12:raise ValueError('Challenger numeric discrepancy')
 budgets=pd.read_csv(OUT/'paired_date_loss_budget.csv');qdelta=pd.read_csv(OUT/'calibration_date_quantile_delta.csv');summary=pd.read_csv(OUT/'paired_date_block_bounds.csv')
 for (t,ca,co),g in budgets.groupby(['track','candidate','control']):
  val=summary[(summary.track==t)&(summary.candidate==ca)&(summary.control==co)].delta.iloc[0]
  if len(g)!=52 or abs(g.signed_loss_budget_contribution.sum()-val)>1e-12:raise ValueError('Date loss budget not retained')
 for t,g in qdelta.groupby('track'):
  val=summary[(summary.track==t)&(summary.candidate=='SIA_calibrated_v34')].delta.iloc[0]
  if len(g)!=260 or abs(g.total_loss_contribution.sum()-val)>1e-12:raise ValueError('Calibrated date-quantile budget')
 ranks=pd.read_csv(OUT/'seed_rank_stability.csv');rank_checks=0
 for row in ranks.itertuples():
  a=table.loc[(row.track,row.model,row.decision_time)].xs(row.seed_a,level='seed').sort_index();b=table.loc[(row.track,row.model,row.decision_time)].xs(row.seed_b,level='seed').sort_index();A=a.q50.sort_values(ascending=False,kind='stable').index;B=b.q50.sort_values(ascending=False,kind='stable').index
  if (A[0]!=B[0])!=row.top1_switch or abs(len(set(A[:2])&set(B[:2]))/len(set(A[:2])|set(B[:2]))-row.top2_jaccard)>1e-12:raise ValueError('Seed rank/tie check failed')
  rank_checks+=1
 age=pd.read_csv(OUT/'outer_source_age_clock_audit.csv');thresholds=readj(R/'reports/model_risk_v3/threshold_candidates.json');agechecks=0
 for row in age.itertuples():
  year=2020 if row.track=='Main-A' else 2022;p=thresholds[row.track+':'+str(year)][row.source+'_release_age_feature_warning']
  if dt(row.train_cutoff)>=dt(row.decision_time) or dt(row.maximum_feature_dependency)>dt(row.decision_time) or p['outer_outcomes_used'] or row.age_days<0:raise ValueError('Source age chronology')
  if abs(row.threshold95_days-p['thresholds']['0.95'])>1e-12 or abs(row.threshold99_days-p['thresholds']['0.99'])>1e-12:raise ValueError('Age candidate altered')
  if (row.age_days>row.threshold95_days)!=row.exceeds95 or (row.age_days>row.threshold99_days)!=row.exceeds99:raise ValueError('Age breach arithmetic')
  agechecks+=1
 control=pd.read_csv(OUT/'model_risk_control_map_current.csv');assert len(control)==32 and control.control_id.nunique()==32
 if len(cur)!=7488 or len(ranks)!=1560:raise ValueError('Diagnostic rows missing')
 codefiles={str(p.relative_to(R)):sha(p) for p in (R/'model_risk_v38').iterdir() if p.is_file()};result={'passed':True,'scope':'Full v38 developer separate-implementation audit, not external independent governance','initial_independent_sha256':sha(OUT/'independent_validation.json'),'two_numeric_routines_verified':True,'monitor_records_recomputed':prior['monitor_records_independently_recomputed'],'challenger_rows_recomputed':len(cur),'maximum_challenger_delta_difference':maxdelta,'seed_rank_pairs_recomputed':rank_checks,'source_age_candidates_verified':agechecks,'OOF_fits_model_checkpoints_reverified':fitcount,'v34_reports_reverified':len(idx['files']),'v37_archive_payloads_verified':len(v37['artifacts']),'v37_documented_archive_files':handoff['archived_files'],'original_files_verified':len(protection['original_files']),'dev_files_verified':len(protection['dev_files']),'frozen_v2_hash':source_hash(),'original_source_hash':code_hash(ORIGINAL),'plan_sha256':sha(OUT/'frozen_plan.json'),'v38_code_hash':digest(codefiles),'code_files':codefiles,'tests':tests,'original_compute_ledgers_accessed':False,'prior_ledger_proof':'reports/evidence_procurement_v37/independent.json (if named differently use handoff.independent_receipt); historical proof not reopened','new_training':0,'new_GPU_inference':0,'reserved_data_access':False}
 # Exact existing ledger proof path from immutable handoff instead of assuming a filename.
 result['prior_ledger_proof']=handoff['independent_receipt']
 writej('independent_validation_final.json',result)
 return result

def main():
 result=validate_final();progress('FINAL_PASS_BLOCKED_GATES_RETAINED',{'tests':result['tests']['counts'],'audit':'COMPLETED','Tier_A':0,'P1':'0/27088','BAR_OOF':'BLOCKED','OOF_source_clocks':'BLOCKED','final':'SEALED'})
 # The last live event is part of the immutable archive; no background monitors remain.
 files={str(p.relative_to(R)):p.read_bytes() for base in [R/'model_risk_v38',OUT] for p in base.iterdir() if p.is_file() and p.name!='final_completion.json'}
 index={rel:hashlib.sha256(value).hexdigest() for rel,value in files.items()};bid=digest(index);payload={f'{i:03d}_{Path(rel).name}':value for i,(rel,value) in enumerate(sorted(files.items()))};payload['archive_index.json']={'files':index,'flattened':{f'{i:03d}_{Path(rel).name}':rel for i,(rel,_) in enumerate(sorted(files.items()))}}
 relative='artifacts/model_risk_v38/completion/'+bid;archive=commit_bundle(T/relative,payload,{'scope':CLAIM,'v38_code_hash':result['v38_code_hash'],'plan_sha256':result['plan_sha256'],'new_training':0,'reserved_access':False});validate_bundle(T/relative)
 final={'state':'COMPLETED_SCOPED_DIAGNOSTICS_WITH_SCIENTIFIC_BLOCKERS','claim':CLAIM,'code_hash':result['v38_code_hash'],'plan_hash':result['plan_sha256'],'files':index,'relative_bundle':relative,'archive_receipt_sha256':sha(T/relative/'receipt.json'),'archive_payload_count':len(archive['artifacts']),'outer_rows':12480,'OOF_rows':12168,'OOF_dates':{'Main-A':441,'Nested-B':66},'monitor_replay_records':4737,'source_age_dry_run_rows':1976,'new_model_fits':0,'new_calibrator_fits':0,'new_GPU_training':0,'new_GPU_inference':0,'new_downloads':0,'tests':result['tests']['counts'],'new_test_cases':75,'prior_pure_tests':23,'original_files_preserved':357,'dev_files_preserved':577,'v2_source_hash':FROZEN,'original_ledgers_opened':False,'previous_studies_intact':True,'controls_mapped':32,'Tier_A':0,'P1_qualified':0,'P1_denominator':27088,'future_independent_dates':0,'reserved_final':'SEALED','strict_final_claims_permitted':False,'whole_project_implementation_complete':False,'reserved_evaluation_complete':False,'independent_validation':'independent_validation_final.json','historical_initial85pass_preserved':True,'report':'V38_INCREMENTAL_MODEL_RISK_AND_CALIBRATION_REPORT.md','no_background_job_started':True,'blockers':['BAR-specific OOF unavailable','OOF source-age/availability dependency clocks absent','Nested52lookback26window unsupported; q99 tail expected .66dates','saved log_volume absent','authentic TierA original-public evidence missing','P1 corporate action/open/pay/cash evidence unqualified','independent governance/operational freeze/future cohort unavailable']}
 writej('final_completion.json',final);print(json.dumps({k:v for k,v in final.items() if k not in ['files','blockers']},indent=2),flush=True)
if __name__=='__main__':main()
