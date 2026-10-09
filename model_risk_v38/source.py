"""v38 CPU-only saved-forecast audit. Never opens original ledgers or canonical data."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev')
ORIGINAL=R.with_name('SignalForge-QX')
T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering')
OUT=R/'reports/model_risk_v38'
Q=np.array([.05,.10,.50,.90,.95]); QC=['q05','q10','q50','q90','q95']
MODELS=['original_RGMF','strong_I0','SIA_v2','BAR','SIA_calibrated_v34']
KEY=['track','decision_time','asset','seed']; CLAIM='EXPLORATORY_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION'
FROZEN='471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def writej(name,value):
 value=json.loads(json.dumps(value,allow_nan=False,default=str))
 if (OUT/name).exists():
  if readj(OUT/name)!=value:raise ValueError('Immutable new receipt conflict: '+name)
  return
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def csv(name,frame):
 content=frame.to_csv(index=False)
 if (OUT/name).exists():
  if (OUT/name).read_text()!=content:raise ValueError('Immutable new CSV conflict: '+name)
  return
 with (OUT/name).open('x') as f:f.write(content)
def progress(stage,detail):
 from datetime import datetime,timezone
 p=OUT/'live_progress.json';d=readj(p) if p.exists() else {'events':[],'training':0,'downloads':0,'claim':CLAIM}
 d['events'].append({'at':datetime.now(timezone.utc).isoformat(),'stage':stage,'detail':detail});p.write_text(json.dumps(d,indent=2));print(stage,detail,flush=True)
def source_hash():
 from signalforge.runtime import code_hash
 return code_hash(R)
def preservation():
 baseline=readj(R/'reports/evidence_procurement_v37/preflight.json')['protected_files']
 h=readj(R/'reports/evidence_procurement_v37/handoff.json');baseline.update(h['files']);baseline['reports/evidence_procurement_v37/handoff.json']=sha(R/'reports/evidence_procurement_v37/handoff.json')
 for rel,hsh in baseline.items():
  if sha(R/rel)!=hsh:raise ValueError('Protected dev hash changed: '+rel)
 old={k.replace(chr(92),'/'):v for k,v in readj(R/'reports/v33_isolation_receipt.json')['copied_files'].items()}
 for rel,hsh in old.items():
  if sha(ORIGINAL/rel)!=hsh:raise ValueError('Original hash changed: '+rel)
 assert source_hash()==FROZEN
 return {'dev_files':baseline,'original_files':old,'frozen_v2_hash':FROZEN,'original_ledger_access':'NOT_OPENED; prior v37 preservation receipt reused per explicit v38 restriction'}
def guard_path(path):
 p=Path(path).resolve()
 allowed=[R/'reports/model_risk_v3',R/'reports/calibration_v34',OUT]
 if not any(p.is_relative_to(a) for a in allowed):raise PermissionError('Only registered development evidence paths')
 if any(s in str(p).lower() for s in ['ledger','reserved','final_labels']):raise PermissionError('Protected scope')
 return p
def aware(values):
 for v in values:
  if pd.Timestamp(v).tzinfo is None:raise ValueError('Timezone required')
 return pd.to_datetime(values,utc=True)
def validate_predictions(df,outer=True):
 if not set(QC+KEY+['target','scale']).issubset(df):raise ValueError('Missing schema')
 df=df.copy();df['decision_time']=aware(df.decision_time)
 if df.decision_time.dt.year.ge(2024).any():raise PermissionError('Reserved date')
 if set(df.track)!={'Main-A','Nested-B'}:raise ValueError('Wrong tracks')
 x=df[QC+['target','scale']].to_numpy(float)
 if not np.isfinite(x).all() or (df.scale<=0).any():raise ValueError('Invalid numeric')
 if (np.diff(x[:,:5],axis=1)<0).any():raise ValueError('Crossing')
 ids=KEY+(['model'] if outer else [])
 if df.duplicated(ids).any():raise ValueError('Duplicate keys')
 if set(df.seed)!={11,37,71}:raise ValueError('Seed mismatch')
 if outer:
  if set(df.model)!=set(MODELS) or len(df)!=12480:raise ValueError('Incomplete models/rows')
  for (t,m),g in df.groupby(['track','model']):
   if len(g)!=1248 or g.decision_time.nunique()!=52 or g.asset.nunique()!=8:raise ValueError('Panel mismatch')
   expected=2020 if t=='Main-A' else 2022
   if set(g.decision_time.dt.year)!={expected}:raise ValueError('Wrong outer year')
  base=df[df.model=='strong_I0'].set_index(KEY).sort_index()
  for m in MODELS:
   other=df[df.model==m].set_index(KEY).sort_index()
   if not base.index.equals(other.index):raise ValueError('Unmatched grid')
   for col in ['target','scale','fold_id','normalizer_id','target_id','source_id']:
    if col in base and not np.array_equal(base[col],other[col]):raise ValueError('Unmatched '+col)
 return df

def freeze():
 p=preservation();writej('preservation_start.json',p)
 plan={'identity':'sgqx-v38-posthoc-model-risk-1','claim':CLAIM,'models':MODELS,'tracks':{'Main-A':2020,'Nested-B':2022},'quantiles':Q.tolist(),'seeds':[11,37,71],
 'slices':['asset','seed','realized_vol_20_regime','return_1w_regime','return_4w_regime','OUTCOME_CONDITIONAL_magnitude','availability_count','any_source_OOD','cftc_available','eia_available','macro_available','nport_available','cftc_age_warning99','eia_age_warning99','macro_age_warning99','nport_age_warning99'],
 'minimum_slice_distinct_dates':13,'bootstrap':{'date_block_sizes':[4,8,13],'draws':2000,'seed':38001,'method':'circular moving block; descriptive only'},
 'influence_top_k':[1,3,5],'rolling_date_windows':[13,26],'acf_lags':[1,2,3,4,8],
 'monitor':{'family':'SIA_v2','quantile':.95,'lookbacks':[26,52],'minimum_dates':'equal lookback','windows':[13,26],'statistics':['normalized_median_absolute_error','interval90_miss_fraction','normalized_interval90_width'],
 'threshold_rule':'q95 of last L prior rolling W date statistics, each fully mature at candidate time; requires L+W-1 prior dates; exclude current decision; current trailing W includes current mature date',
 'lagging_computation_time':'maximum label_available_at of current date; prior observations must have strictly earlier decision and label_available_at <= this time',
 'leading_computation_time':'current decision; widths require only earlier forecast dates, no labels used','q99_nested':'UNRELIABLE_TAIL expected .66 independent dates','dry_run':True},
 'no_training':True,'no_calibrator_fit':True,'no_downloads':True,'source_clock':'Tier B reconstructed; never Tier A',
 'signal_mappings':['top1','top2','linear_rank'],'asset_cap':.25,'commodity_cap':.35,'gross_cap':1.,'absence_log_volume':'BLOCKED_SAVED_FEATURE_ABSENT','no_P1_or_final':True}
 writej('frozen_plan.json',plan)
 paths=[]
 for sub in ['reports/model_risk_v3','reports/calibration_v34']:
  paths.extend(p for p in (R/sub).iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md'])
 idx={'plan_sha256':sha(OUT/'frozen_plan.json'),'inputs':{str(p.relative_to(R)):sha(p) for p in paths},'original_ledgers_opened':False}
 writej('source_index.json',idx);writej('input_source_index.json',idx);progress('PREFLIGHT_PASS',{'protected_dev':len(p['dev_files']),'original':len(p['original_files']),'plan_hash':idx['plan_sha256']})
if __name__=='__main__':freeze()
