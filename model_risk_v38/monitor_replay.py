"""Mature-label causal monitor replay. No model/calibrator fits and no operational alerts."""
from .source import *
from .diagnostics import enrich,date_series
from datetime import datetime,timezone

def mature_rows(frame,at,decision_before):
 a=pd.Timestamp(at);d=pd.Timestamp(decision_before)
 if a.tzinfo is None or d.tzinfo is None:raise ValueError('Aware replay cutoffs required')
 return frame[(frame.label_available_at<=a)&(frame.decision_time<d)]
def validate_oof(df):
 d=validate_predictions(df,outer=False)
 for col in ['label_end','label_available_at']:d[col]=aware(d[col])
 if set(d.role)!={'OOF'}:raise ValueError('Not genuine OOF role')
 if (d.label_end<d.decision_time).any() or (d.label_available_at<d.label_end).any():raise ValueError('Invalid label chronology')
 for t,g in d.groupby('track'):
  cutoff=pd.Timestamp('2020-01-01T00:00:00Z' if t=='Main-A' else '2022-01-01T00:00:00Z');dates=441 if t=='Main-A' else 66
  if (g.label_available_at>=cutoff).any() or (g.decision_time>=cutoff).any():raise ValueError('OOF outer leakage')
  if g.decision_time.nunique()!=dates or len(g)!=dates*24:raise ValueError('OOF support mismatch')
  if (g.groupby('decision_time').size()!=24).any():raise ValueError('OOF incomplete date')
 return d

def require_mature(labels,at):
 a=pd.Timestamp(at)
 if a.tzinfo is None:raise ValueError('Aware maturity cutoff required')
 if any(pd.Timestamp(x)>a for x in labels):raise ValueError('Immature label use')

def event_guard(pred,target=None,feature_at=None,decision=None,available_at=None,at=None,age=None,max_age=None,expected_hash=None,actual_hash=None,checkpoint_ok=True,signer=False,deployment=False):
 x=np.asarray(pred,float)
 if not np.isfinite(x).all():raise ValueError('Nonfinite')
 if x.shape[-1]!=5 or (np.diff(x,axis=-1)<0).any():raise ValueError('Crossing/schema')
 if target is not None and not np.isfinite(target):raise ValueError('Invalid target')
 if feature_at is not None:
  a=pd.Timestamp(feature_at);b=pd.Timestamp(decision)
  if a.tzinfo is None or b.tzinfo is None or a>b:raise ValueError('Future/unverified feature clock')
 if available_at is not None:require_mature([available_at],at)
 if age is not None and (not np.isfinite(age) or age<0 or (max_age is not None and age>max_age)):raise ValueError('Invalid/stale source')
 if expected_hash is not None and expected_hash!=actual_hash:raise ValueError('Source hash mismatch')
 if not checkpoint_ok:raise ValueError('Corrupt checkpoint')
 if deployment and not signer:raise PermissionError('Absent independent signer; no deployment')
 return True

def threshold_for(history,lookback,window):
 x=np.asarray(history,float)
 if not np.isfinite(x).all():raise ValueError('Invalid monitor statistic')
 n=lookback+window-1
 if len(x)<n:return None
 h=x[-n:];rolling=np.convolve(h,np.ones(window)/window,'valid')
 assert len(rolling)==lookback
 return float(np.quantile(rolling,.95))

def replay(oof):
 records=[];summary=[];support={};oo=enrich(oof)
 for t,g in oo.groupby('track'):
  daily=date_series(g,['normalized_abs_error','coverage90','normalized_width90']);daily['interval90_miss_fraction']=1-daily.coverage90
  daily['available_at']=g.groupby('decision_time').label_available_at.max().reindex(daily.index)
  for statistic,col,leading in [('normalized_median_absolute_error','normalized_abs_error',False),('interval90_miss_fraction','interval90_miss_fraction',False),('normalized_interval90_width','normalized_width90',True)]:
   for L in [26,52]:
    for W in [13,26]:
     rows=[]
     for date,cur in daily.iterrows():
      time=date if leading else cur.available_at
      hist=daily[daily.index<date]
      if not leading:hist=hist[hist.available_at<=time]
      thr=threshold_for(hist[col].to_numpy(),L,W)
      if thr is None:continue
      current=pd.concat([hist[[col]],pd.DataFrame({col:[cur[col]]},index=[date])]).tail(W)
      if len(current)!=W:raise ValueError('Window insufficient')
      value=float(current[col].mean());exceed=value>thr
      row={'track':t,'model':'SIA_v2','statistic':statistic,'leading':leading,'lookback':L,'window':W,'decision_time':date,'threshold_computation_time':time,'max_historical_decision_time':hist.index[-1],'max_historical_label_available_at':None if leading else hist.available_at.max(),'historical_distinct_dates':len(hist),'threshold_distribution_dates':L,'threshold':thr,'current_value':value,'exceeded':exceed,'forecast_to_maturity_days':float((cur.available_at-date).total_seconds()/86400),'label_used':not leading,'source_clock':'Tier B reconstructed','dry_run':True,'actual_computed_at':datetime.now(timezone.utc).isoformat()};records.append(row);rows.append(row)
     s={'track':t,'statistic':statistic,'leading':leading,'lookback':L,'window':W,'required_prior_dates':L+W-1,'OOF_dates':len(daily),'eligible_dates':len(rows),'exceedance_dates':sum(r['exceeded'] for r in rows),'state':'DRY_RUN_DESCRIPTIVE' if rows else 'UNDERPOWERED_NO_ELIGIBLE_DATE','first_feasible_decision':str(rows[0]['decision_time']) if rows else None,'first_computation_time':str(rows[0]['threshold_computation_time']) if rows else None}
     if rows:
      flags=[r['exceeded'] for r in rows];runs=[];run=0
      for flag in flags:
       if flag:run+=1
       elif run:runs.append(run);run=0
      if run:runs.append(run)
      s.update(exceedance_fraction=float(np.mean(flags)),alert_runs=len(runs),maximum_run=max(runs,default=0),threshold_min=min(r['threshold'] for r in rows),threshold_max=max(r['threshold'] for r in rows),median_maturity_lag_days=float(np.median([r['forecast_to_maturity_days'] for r in rows])))
     summary.append(s)
  support[t]={'OOF_dates':len(daily),'rows':len(g),'q99_expected_extreme_tail_dates':len(daily)*.01,'q99_state':'UNRELIABLE_TAIL' if len(daily)*.01<1 else 'LOW_TAIL_SUPPORT_DESCRIPTIVE_ONLY','source_age_OOF_state':'BLOCKED_UNVERIFIED_CLOCK: genuine OOF CSV contains no source-age/availability dependency clocks','mature_label_clock':'reconstructed Tier B only'}
 df=pd.DataFrame(records);csv('training_only_monitor_replay.csv',df);csv('threshold_monitor_replay.csv',df);csv('thresholds_by_date.csv',df[['track','statistic','lookback','window','decision_time','threshold_computation_time','threshold','historical_distinct_dates']]);csv('monitor_summary.csv',pd.DataFrame(summary))
 writej('threshold_support.json',support)
 baseline=readj(R/'reports/model_risk_v3/threshold_candidates.json')
 catalog={'SIA':'genuine mature OOF causal replay, thresholds not operationally frozen','strong_I0':{t:v['fixed_baseline_OOF_error'] for t,v in baseline.items()},'BAR':'BLOCKED_OOF_UNAVAILABLE; baseline thresholds not substituted','original_RGMF':'BLOCKED_OOF_UNAVAILABLE','leading':['normalized_interval90_width'],'lagging':['normalized_median_absolute_error','interval90_miss_fraction'],'source_age_OOF':'BLOCKED_UNVERIFIED_CLOCK','hard_stops':['schema','nonfinite','crossing','source/checkpoint hash','duplicate key','immature label','reserved date'],'notifications_sent':0}
 writej('leading_lagging_monitor_catalog.json',catalog);writej('alert_limitations.json',{'state':'HISTORICAL_DRY_RUN_NOT_FORWARD_FORECAST','source_clock':'Tier B reconstructed','no_false_positive_rate_claim':True,'no_event_truth':True,'no_operational_freeze':True,'owners_appointed':False,'nested_q99':'66 dates => .66 expected tail dates; UNRELIABLE_TAIL','no_new_calibrator_fits':True,'only_prior_mature_labels':True,'threshold_date_samples_overlap':'rolling windows and market regimes dependent; exceedance rates descriptive, not independent Bernoulli'})
 progress('MATURE_OOF_REPLAY_PASS',{'actual_rows':len(df),'OOF_dates':support,'eligible_summary_rows':len(summary)})
 return df
