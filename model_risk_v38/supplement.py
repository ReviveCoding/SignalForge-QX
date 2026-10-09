"""User-requested complete date/rank/clock/control support supplements, no fitting."""
from .source import *
from .diagnostics import enrich,paired,PAIRS,block_bounds,date_series
from scipy.stats import spearmanr

def run():
 d=enrich(validate_predictions(pd.read_csv(R/'reports/calibration_v34/outer_forecasts.csv')));ex=pd.read_csv(R/'reports/model_risk_v3/ex_ante_features.csv');ex.decision_time=aware(ex.decision_time);d=d.merge(ex,on=['track','decision_time','asset'],validate='many_to_one',suffixes=('','_exante'))
 dateparts=[];qparts=[];rankseed=[];disagreement=[]
 for t,g in d.groupby('track'):
  for ca,co in PAIRS:
   x=paired(g,ca,co)
   for date,h in x.groupby('decision_time'):
    dateparts.append({'track':t,'candidate':ca,'control':co,'decision_time':date,'rows':len(h),'delta':h.delta.mean(),'signed_loss_budget_contribution':h.delta.sum()/len(x),'single_date':'UNDERPOWERED_EVENT_ATTRIBUTION_NOT_INDEPENDENT_CONFIRMATION'})
    if ca=='SIA_calibrated_v34':
     for c in QC:qparts.append({'track':t,'decision_time':date,'quantile':c,'delta':h['delta_'+c].mean(),'total_loss_contribution':h['delta_'+c].sum()/len(x)/5})
   for _,row in x.iterrows():
    signA=np.sign(row.q50);signB=np.sign(row.control_q50);truth=np.sign(row.target)
    disagreement.append({'track':t,'candidate':ca,'control':co,'decision_time':row.decision_time,'asset':row.asset,'seed':row.seed,'delta':row.delta,'raw_median_change':row.q50-row.control_q50,'median_change_normalized':(row.q50-row.control_q50)/row.scale,'baseline_abs_error_normalized':abs(row.target-row.control_q50)/row.scale,'candidate_abs_error_normalized':abs(row.target-row.q50)/row.scale,'corrects_loss':row.delta<0,'hurts_loss':row.delta>0,'control_wrong_candidate_right':signB!=truth and signA==truth,'control_right_candidate_wrong':signB==truth and signA!=truth,'both_wrong':signA!=truth and signB!=truth,'sign_disagreement':signA!=signB,'near_zero_control':abs(row.control_q50/row.scale)<1e-3,'realized_vol_20_regime':row.realized_vol_20_regime,'return_4w_regime':row.return_4w_regime,'availability_count':row.availability_count,'any_source_OOD':row.any_source_OOD})
  for (m,date),h in g.groupby(['model','decision_time']):
   for s1,s2 in [(11,37),(11,71),(37,71)]:
    a=h[h.seed==s1].sort_values('asset');b=h[h.seed==s2].sort_values('asset');v=a.q50.to_numpy();w=b.q50.to_numpy();orderA=sorted(range(8),key=lambda i:(-v[i],a.asset.iloc[i]));orderB=sorted(range(8),key=lambda i:(-w[i],a.asset.iloc[i]));A=set(orderA[:2]);B=set(orderB[:2]);rankseed.append({'track':t,'model':m,'decision_time':date,'seed_a':s1,'seed_b':s2,'spearman':float(spearmanr(v,w).statistic) if np.ptp(v)>0 and np.ptp(w)>0 else np.nan,'top1_switch':orderA[0]!=orderB[0],'top2_jaccard':len(A&B)/len(A|B),'ties_a':8-len(np.unique(v)),'ties_b':8-len(np.unique(w)),'tie_rule':'descending q50 then lexical asset','high_vol_assets':int((a.realized_vol_20_regime=='HIGH').sum()),'duplicate_baseline':m=='strong_I0'})
 csv('paired_date_loss_budget.csv',pd.DataFrame(dateparts));csv('calibration_date_quantile_delta.csv',pd.DataFrame(qparts));csv('seed_rank_stability.csv',pd.DataFrame(rankseed));csv('challenger_row_disagreement.csv',pd.DataFrame(disagreement))
 replay=pd.read_csv(OUT/'training_only_monitor_replay.csv');bounds=[]
 for keys,h in replay.groupby(['track','statistic','lookback','window']):
  for block in [4,8,13]:
   rec=dict(zip(['track','statistic','lookback','window'],keys));rec.update(block=block,eligible_dates=len(h),exceedance_fraction=float(h.exceeded.mean()),claim='DEPENDENT_DATE_DIAGNOSTIC_NOT_FALSE_POSITIVE_RATE')
   if len(h)>=block:lo,hi=block_bounds(h.exceeded.to_numpy(float),block);rec.update(lower_descriptive=lo,upper_descriptive=hi,state='DESCRIPTIVE_ONLY',draws=2000)
   else:rec.update(state='UNDERPOWERED_FEWER_DATES_THAN_BLOCK',draws=0)
   bounds.append(rec)
 csv('monitor_date_block_bounds.csv',pd.DataFrame(bounds))
 thresholds=readj(R/'reports/model_risk_v3/threshold_candidates.json');clock=readj(R/'reports/calibration_v34/monitor_clock_validation.json');clockdates={}
 for row in clock['rows']:
  key=(row['track'],pd.Timestamp(row['date']));dependency=pd.Timestamp(row['max_feature_dependency_at']);clockdates[key]=max(dependency,clockdates.get(key,dependency))
 age_rows=[]
 for _,row in ex.iterrows():
  t=row.track;date=row.decision_time
  for source in ['cftc','eia','macro','nport']:
   column=source+'_release_age_max_days';candidate=thresholds[t+':'+str(row.year)].get(source+'_release_age_feature_warning')
   if candidate is None or column not in row or pd.isna(row[column]):continue
   cutoff=pd.Timestamp(candidate['cutoff']);dep=clockdates[(t,date)]
   if cutoff>=date or candidate['outer_outcomes_used'] or dep>date:raise ValueError('Age threshold/provenance leakage')
   age=float(row[column]);state='REGISTERED_TIER_B_DRY_RUN' if age>=0 else 'INVALID_NEGATIVE_AGE'
   if age<0:raise ValueError('Negative source age')
   age_rows.append({'track':t,'decision_time':date,'asset':row.asset,'source':source,'available':row[source+'_available'],'age_days':age,'threshold95_days':candidate['thresholds']['0.95'],'threshold99_days':candidate['thresholds']['0.99'],'exceeds95':age>candidate['thresholds']['0.95'],'exceeds99':age>candidate['thresholds']['0.99'],'train_cutoff':cutoff,'maximum_feature_dependency':dep,'state':state,'clock':'reconstructed Tier B, not first public','alert_sent':False})
 csv('outer_source_age_clock_audit.csv',pd.DataFrame(age_rows))
 old=readj(R/'reports/model_risk_v3/parameter_stability.json');writej('parameter_evidence_reuse.json',{'source_sha256':sha(R/'reports/model_risk_v3/parameter_stability.json'),'prior_top_level_keys':list(old),'new_parameter_perturbations':0,'claims':'Prior norm/cosine descriptive and not permutation invariant; new functional forecasts/ranks preferred','existing_source_perturbations':21,'CPU_CUDA_difference_preserved':4.3479e-6,'no_causal_source_attribution':True})
 progress('SUPPLEMENTAL_DIAGNOSTICS_PASS',{'paired_dates':len(dateparts),'seed_rank_pairs':len(rankseed),'challenger_asset_rows':len(disagreement),'source_age_dry_run_rows':len(age_rows),'monitor_block_bound_rows':len(bounds)})
if __name__=='__main__':run()
