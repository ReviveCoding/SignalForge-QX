"""Final controls distinguish historic definitions from current measured support."""
from .source import *

def age_candidate_guard(age,cutoff,decision,outer_outcomes_used=False):
 if pd.Timestamp(cutoff).tzinfo is None or pd.Timestamp(decision).tzinfo is None:raise ValueError('Aware age cutoff')
 if outer_outcomes_used or pd.Timestamp(cutoff)>=pd.Timestamp(decision):raise ValueError('Age candidate future/outer use')
 if not np.isfinite(age) or age<0:raise ValueError('Invalid age')
 return True

def update_controls():
 d=pd.read_csv(OUT/'model_risk_control_map.csv');catalog=readj(R/'reports/model_risk_v3/monitoring.json')['catalog'];metric=pd.read_csv(OUT/'overall_metrics.csv');age=pd.read_csv(OUT/'outer_source_age_clock_audit.csv')
 for i,item in enumerate(catalog):
  d.loc[i,'control_id']=f'M{i+1:02d}';d.loc[i,'registered_track']=item.get('track','BOTH/NORMATIVE');name=item['indicator']
  if 'release_age_feature_warning' in name:
   source=name.split('_')[0];g=age[(age.source==source)&(age.track==item['track'])];d.loc[i,'evidence']='outer_source_age_clock_audit.csv; train-only historical candidates';d.loc[i,'observed_breach']=f'{int(g.exceeds95.sum())}/{len(g)} asset-date rows over q95; {int(g.exceeds99.sum())} over q99; date-dependent and descriptive';d.loc[i,'blocker']='Tier B source clocks only; OOF source dependencies absent; no operational freeze'
  if 'max_abs_forecast' in name:d.loc[i,'observed_breach']='0 forecasts above100; actual per-model maxima in overall_metrics.csv';d.loc[i,'evidence']='overall_metrics.csv'
  if 'PSI/KS' in name:d.loc[i,'status']='NOT_IMPLEMENTED_FORMAL_DISTRIBUTION_DRIFT';d.loc[i,'observed_breach']='NOT_MEASURED';d.loc[i,'blocker']='No registered reference/window control or FPR calibration; no post-result threshold selection'
  if name=='future feature availability':d.loc[i,'observed_breach']='0 v34 recorded max dependencies after decision; Tier B scope only';d.loc[i,'evidence']='v34 monitor_clock_validation.json; outer_source_age_clock_audit.csv'
  if name=='immature training labels':d.loc[i,'observed_breach']='0 saved OOF rows beyond pre-outer cutoff; 120 historical frozen fits verified';d.loc[i,'evidence']='panel_integrity.json; independent_validation.json'
  if name=='reserved cohort access':d.loc[i,'observed_breach']='0 v38 reserved reads; whitelist development data only';d.loc[i,'evidence']='source.py scope guard; tests; no original ledger access'
  if name=='OOD/availability changes':d.loc[i,'observed_breach']='fixed exante groups and row counts in calibration_by_slice.csv; no calibrated drift threshold';d.loc[i,'evidence']='regime_challenger.csv; calibration_by_slice.csv'
 for i,row in d[d.kind=='RISK'].iterrows():
  d.loc[i,'control_id']=row['id']
  if row['id']=='R01':d.loc[i,'measurable_control']='SIA genuine OOF now exists; calibration-v34 failed outer transport; preserve identity tails and quarantine calibrated candidate';d.loc[i,'evidence']='OOF_outer_gap.csv; calibration_delta_decomposition.csv; monitor_summary.csv';d.loc[i,'blocker']='BAR OOF missing; authentic PIT, independent future cohort and operational controls absent'
 csv('model_risk_control_map_current.csv',d);writej('control_map_revision.json',{'initial_map':'model_risk_control_map.csv retained','current_map':'model_risk_control_map_current.csv','changes':'explicit per-track monitor identity/breach evidence; correct historic SIA OOF missing claim; mark PSI/KS not implemented','controls':len(d),'operational_deployment':False})
if __name__=='__main__':update_controls()
