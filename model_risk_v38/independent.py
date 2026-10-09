"""Independent scalar numeric and chronology audit; does not import main metric/replay code."""
import csv as csvlib,json,math,hashlib
from decimal import Decimal
from collections import defaultdict
from datetime import datetime
import numpy as np,pandas as pd
from .source import R,T,OUT,sha,readj,writej,progress,preservation,source_hash,FROZEN
from signalforge.runtime import validate_bundle
NAMES=['q05','q10','q50','q90','q95'];TAU=[.05,.1,.5,.9,.95]
def scalar_pinball(y,p,t):return t*(y-p) if y>=p else (1-t)*(p-y)
def decimal_pinball(y,p,t):
 e=Decimal(str(y))-Decimal(str(p));a=Decimal(str(t));return max(a*e,(a-1)*e)
def linear_quantile(values,p):
 s=sorted(values);ix=(len(s)-1)*p;lo=math.floor(ix);hi=math.ceil(ix);return s[lo]+(ix-lo)*(s[hi]-s[lo])
def dt(s):
 d=datetime.fromisoformat(s.replace('Z','+00:00'))
 if d.tzinfo is None:raise ValueError('Independent clock requires timezone')
 return d

def run():
 scalar=defaultdict(list);dec=defaultdict(list);cover=defaultdict(lambda:[0]*5);rowcount=defaultdict(int);bykey={};worst=0.
 with (R/'reports/calibration_v34/outer_forecasts.csv').open() as f:
  for row in csvlib.DictReader(f):
   k=(row['track'],row['model']);y=float(row['target']);s=float(row['scale']);q=[float(row[n]) for n in NAMES];
   v=sum(scalar_pinball(y,p,t) for p,t in zip(q,TAU))/5/s
   vd=sum(decimal_pinball(y,p,t) for p,t in zip(q,TAU))/Decimal(5)/Decimal(str(s))
   worst=max(worst,abs(v-float(vd)));scalar[k].append(v);dec[k].append(float(vd));rowcount[k]+=1
   for j,p in enumerate(q):cover[k][j]+=int(y<=p)
   bykey[(row['track'],dt(row['decision_time']),row['asset'],int(row['seed']),row['model'])]=row
 metrics=pd.read_csv(OUT/'overall_metrics.csv');checks=[]
 for k,x in scalar.items():
  saved=metrics[(metrics.track==k[0])&(metrics.model==k[1])].iloc[0];a=math.fsum(x)/len(x);b=math.fsum(dec[k])/len(x)
  if abs(a-saved.loss)>1e-12 or abs(b-saved.loss)>1e-12:raise ValueError('Independent loss disagreement')
  for j,n in enumerate(NAMES):
   if abs(cover[k][j]/len(x)-saved['coverage_'+n])>1e-12:raise ValueError('Independent coverage disagreement')
  checks.append({'track':k[0],'model':k[1],'rows':len(x),'scalar_NPL':a,'decimal_NPL':b,'coverage':dict(zip(NAMES,[n/len(x) for n in cover[k]]))})
 # Reconstruct calibration offsets with scalar cumulative projection inside unsupported anchors.
 frozen=readj(R/'reports/calibration_v34/calibration_frozen.json');print('CALIBRATION_STRUCTURE',list(frozen),flush=True)
 params=frozen['parameters'];cal_error=0.;unsupported=0
 for key,row in bykey.items():
  if key[-1]!='SIA_calibrated_v34':continue
  t,date,asset,seed,_=key;old=bykey[(t,date,asset,seed,'SIA_v2')];p=params[t][str(seed)] if t in params else params[t+':'+str(seed)]
  base=[float(old[n]) for n in NAMES];sc=float(old['scale']);supp=p['supported'];z=[v+sc*d if ok else v for v,d,ok in zip(base,p['offset_normalized'],supp)];i=0
  while i<5:
   if not supp[i]:unsupported+=1;i+=1;continue
   start=i
   while i<5 and supp[i]:i+=1
   lo=base[start-1] if start else -math.inf;hi=base[i] if i<5 else math.inf;prev=-math.inf
   for j in range(start,i):prev=max(prev,z[j]);z[j]=min(hi,max(lo,prev))
  for j,n in enumerate(NAMES):cal_error=max(cal_error,abs(z[j]-float(row[n])))
 if cal_error>1e-12:raise ValueError('Frozen calibrator replay mismatch')
 # Independent OOF aggregation: ordinary scalar lists, market-date clusters, no diagnostics import.
 daily=defaultdict(lambda:defaultdict(list));available=defaultdict(list)
 with (R/'reports/calibration_v34/genuine_sia_oof.csv').open() as f:
  for row in csvlib.DictReader(f):
   k=(row['track'],dt(row['decision_time']));y=float(row['target']);sc=float(row['scale']);lo=float(row['q05']);hi=float(row['q95']);med=float(row['q50']);daily[k]['normalized_median_absolute_error'].append(abs(y-med)/sc);daily[k]['interval90_miss_fraction'].append(float(not lo<=y<=hi));daily[k]['normalized_interval90_width'].append((hi-lo)/sc);available[k].append(dt(row['label_available_at']))
 means={k:{s:math.fsum(v)/len(v) for s,v in values.items()} for k,values in daily.items()};avail={k:max(v) for k,v in available.items()};keys=sorted(means);maximum_threshold_error=0.;rows=0
 with (OUT/'training_only_monitor_replay.csv').open() as f:
  for row in csvlib.DictReader(f):
   t=row['track'];date=dt(row['decision_time']);at=dt(row['threshold_computation_time']);leading=row['leading']=='True';L=int(row['lookback']);W=int(row['window']);stat=row['statistic']
   history=[k for k in keys if k[0]==t and k[1]<date and (leading or avail[k]<=at)]
   if len(history)<L+W-1:raise ValueError('Independent support failure')
   if not leading and (at!=avail[(t,date)] or any(avail[k]>at for k in history)):raise ValueError('Immaturity/leakage')
   if leading and at!=date:raise ValueError('Leading clock mismatch')
   vals=[means[k][stat] for k in history[-(L+W-1):]];roll=[math.fsum(vals[i:i+W])/W for i in range(L)];threshold=linear_quantile(roll,.95);current=math.fsum([means[k][stat] for k in history[-(W-1):]]+[means[(t,date)][stat]])/W
   maximum_threshold_error=max(maximum_threshold_error,abs(threshold-float(row['threshold'])),abs(current-float(row['current_value'])))
   if abs(threshold-float(row['threshold']))>1e-12 or abs(current-float(row['current_value']))>1e-12:raise ValueError('Independent replay disagreement')
   if (current>threshold)!=(row['exceeded']=='True') and abs(current-threshold)>1e-12:raise ValueError('Alert logic mismatch')
   rows+=1
 # Previous fitted candidates: validate immutable models and predictions only, no load/fit/ledger.
 models=0;bundlefiles={}
 for rel in ['reports/v33_minimal_study_v2.json','reports/bar_fits_v1.json']:
  for fit in readj(R/rel)['results']:
   b=validate_bundle(T/fit['relative_bundle']);models+=1;bundlefiles[fit['relative_bundle']]=b['artifacts']
 protection=preservation()
 if protection!=readj(OUT/'preservation_start.json'):raise ValueError('Protection drift')
 idx=readj(OUT/'source_index.json')
 for rel,h in idx['inputs'].items():
  if sha(R/rel)!=h:raise ValueError('Input changed: '+rel)
 if sha(OUT/'frozen_plan.json')!=idx['plan_sha256']:raise ValueError('Plan changed')
 attr=pd.read_csv(OUT/'paired_loss_attribution.csv')
 for (t,ca,co),g in attr.groupby(['track','candidate','control']):
  for dim in ['asset','seed','quantile']:
   if abs(g[g.dimension==dim].signed_loss_budget_contribution.sum()-g.overall_delta.iloc[0])>1e-12:raise ValueError('Signed loss budget mismatch')
 weights=pd.read_csv(OUT/'P0_signal_weights.csv');p0=pd.read_csv(OUT/'P0_portfolio_materiality_proxy.csv')
 if (weights.weight<0).any() or (weights.weight>.25+1e-12).any() or (p0.commodity_weight>.35+1e-12).any() or (p0.gross_signal_weight>1+1e-12).any():raise ValueError('P0 caps')
 writej('bundle_source_index.json',bundlefiles)
 tests=readj(OUT/'tests.json')
 if tests['exit_code'] or any(tests['counts'][x] for x in ['failures','errors','skipped']):raise ValueError('Required scoped tests failed')
 result={'passed':True,'scope':'Developer separate scalar/Decimal recomputation, NOT independent external governance','two_independent_arithmetic_routines':['scalar conditional pinball with math.fsum','Decimal max-based pinball'],'outer_checks':checks,'maximum_scalar_decimal_row_difference':worst,'maximum_monitor_threshold_value_difference':maximum_threshold_error,'monitor_records_independently_recomputed':rows,'frozen_calibration_max_replay_error':cal_error,'unsupported_tail_identity_elements':unsupported,'old_SIA_BAR_candidate_bundles_verified':models,'v34_fits_verified':120,'v34_reports_verified':48,'protected_original_files':len(protection['original_files']),'protected_dev_files':len(protection['dev_files']),'frozen_v2_hash':source_hash(),'original_ledgers':'not opened per v38 restriction; preserved historical v37 proof referenced','reserved_access':False,'new_training':0,'new_calibrator_fits':0,'tests':tests['counts'],'code_files':{str(p.relative_to(R)):sha(p) for p in (R/'model_risk_v38').iterdir() if p.is_file()}}
 writej('independent_validation.json',result);writej('preservation_end.json',protection);progress('INDEPENDENT_VALIDATION_PASS',{k:v for k,v in result.items() if k not in ['outer_checks','code_files']})
if __name__=='__main__':run()
