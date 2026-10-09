"""Executable incremental v38 audit; old evidence immutable, CPU analytics only."""
from .source import *
from .diagnostics import enrich,analysis
from .monitor_replay import validate_oof,replay
from .portfolio_proxy import portfolio,controls
from signalforge.runtime import validate_bundle

def integrity_and_load():
 before=readj(OUT/'preservation_start.json');now=preservation()
 if before!=now:raise ValueError('Protection manifest changed')
 d=validate_predictions(pd.read_csv(guard_path(R/'reports/calibration_v34/outer_forecasts.csv')))
 old=readj(R/'reports/model_risk_v3/panel_receipt.json');rec=validate_bundle(T/old['relative_bundle'])
 if sha(T/old['relative_bundle']/'receipt.json')!=old['receipt_sha256']:raise ValueError('Old panel receipt changed')
 prior=pd.read_csv(T/old['relative_bundle']/'panel.csv');prior.decision_time=aware(prior.decision_time)
 for m in MODELS[:-1]:
  a=d[d.model==m].set_index(KEY).sort_index();b=prior[prior.model==m].set_index(KEY).sort_index()
  if not a.index.equals(b.index):raise ValueError('Old panel grid mismatch')
  if not np.allclose(a[QC+['target','scale']],b[QC+['target','scale']],rtol=0,atol=1e-12):raise ValueError('Old predictions changed')
 oof=validate_oof(pd.read_csv(guard_path(R/'reports/calibration_v34/genuine_sia_oof.csv')))
 c=readj(R/'reports/calibration_v34/oof_completion.json');fits=c['results']
 if len(fits)!=120 or c['successful_fits']!=120:raise ValueError('Missing OOF fits')
 joined=[]
 for f in fits:
  rec=validate_bundle(T/f['relative_bundle'])
  if not Path(f['checkpoint_path']).is_relative_to(T/'checkpoints/sia_oof_v34') or sha(f['checkpoint_path'])!=f['checkpoint_hash']:raise ValueError('OOF checkpoint corruption')
  if not f['reload_bitwise_equal'] or f['device']!='NVIDIA GeForce RTX 4090 Laptop GPU':raise ValueError('OOF training provenance')
  z=pd.read_csv(T/f['relative_bundle']/'predictions.csv');z['track']=f['track'];z['seed']=f['seed'];joined.append(z)
 j=pd.concat(joined,ignore_index=True);j.decision_time=aware(j.decision_time)
 a=oof.set_index(KEY).sort_index();b=j.set_index(KEY).sort_index()
 if not a.index.equals(b.index) or not np.allclose(a[QC],b[QC],rtol=0,atol=1e-12):raise ValueError('OOF saved predictions differ')
 idx=readj(R/'reports/calibration_v34/final_index.json')
 for rel,h in idx['files'].items():
  if sha(R/rel)!=h:raise ValueError('v34 report changed')
 validate_bundle(T/idx['relative_bundle'])
 h=readj(R/'reports/evidence_procurement_v37/handoff.json');archive=validate_bundle(T/h['relative_bundle'])
 # v37 archived_files counts flattened handoff payloads, including source/output index.
 ex=pd.read_csv(R/'reports/model_risk_v3/ex_ante_features.csv');ex.decision_time=aware(ex.decision_time)
 if ex.duplicated(['track','decision_time','asset']).any() or len(ex)!=832:raise ValueError('Exante grid mismatch')
 d=enrich(d).merge(ex,on=['track','decision_time','asset'],how='left',validate='many_to_one',suffixes=('','_exante'))
 if d.realized_vol_20_regime.isna().any():raise ValueError('Missing exante regimes')
 clock=readj(R/'reports/calibration_v34/monitor_clock_validation.json')
 if not clock['passed'] or not clock['all_feature_dependencies_at_or_before_origin']:raise ValueError('Feature clocks failed')
 result={'passed':True,'outer_rows':len(d),'old_panel_rows':len(prior),'OOF_rows':len(oof),'OOF_unique_dates':oof.groupby('track').decision_time.nunique().to_dict(),'OOF_fits_checkpoints_bundles_verified':120,'OOF_predictions_match_saved_bundles':True,'v34_reports_verified':len(idx['files']),'v37_archive_payloads_verified':len(archive['artifacts']),'old_four_family_predictions_match':True,'all_rows_retained':True,'crossings':0,'nonfinite':0,'original_ledgers_opened':False,'max_dependency_clock':'Tier B existing v34 receipt, not authenticated first public','frozen_v2_hash':source_hash()}
 writej('panel_integrity.json',result);progress('PANEL_INTEGRITY_PASS',result)
 return d,oof

def coverage():
 old=pd.read_csv(R/'reports/model_risk_v3/method_implementation_audit.csv');rows=[]
 for _,x in old.iterrows():rows.append({'analysis_family':x.method,'previous_status':x.actual_status,'v38_action':'READ_ONLY_REUSE','evidence':x.evidence,'boundary':x['limit']})
 additions=[('SIA genuine mature OOF','historical v3 missing; v34 now executed','120 CUDA fits /12168 rows; no new fitting','calibration_v34/oof_completion.json'),('SIA calibration','v34 negative on outer','QUARANTINED; paired quantile/slice damage diagnosis','calibration_delta_decomposition.csv'),('BAR-specific mature OOF','NOT_IMPLEMENTED','BLOCKED_OOF_UNAVAILABLE','leading_lagging_monitor_catalog.json'),('Causal prequential warning threshold replay','NEW','IMPLEMENTED_CPU_SAVED_OOF_NO_FIT','training_only_monitor_replay.csv'),('Five-family integrated reliability','NEW fifth calibrated SIA','date-equal all rows, fixed slices and tail support','calibration_by_slice.csv'),('Signed loss budget / stress influence','NEW incremental','primary unchanged; exclusions counterfactual only','paired_loss_attribution.csv; stress_influence.csv'),('Functional seed / challenger diagnostics','prior four-family now five','no parameter cosine claims','seed_functional_stability.csv; challenger_disagreement.csv'),('P0 materiality concentration','reuse fixed four-family weights; fifth extension','NO P1 PnL / orders','P0_portfolio_materiality_proxy.csv'),('21 source perturbations','ALREADY_EXECUTED_CPU','REUSED_NOT_RERUN; noncausal CPU/CUDA discrepancy4.3479e-6','reports/model_risk_v3/source_ablation.csv'),('OOF source-age monitors','OOF clock/age columns absent','BLOCKED_UNVERIFIED_CLOCK','alert_limitations.json'),('log_volume slices','saved exante feature absent','BLOCKED_SAVED_FEATURE_ABSENT','ex_ante_features.csv'),('Strict PIT / P1 / final / future','external missing evidence','BLOCKED TierA0 /P1 0of27088 /future0 /final sealed','reports/evidence_procurement_v37/handoff.json'),('ABC / HPO / SSL / 36 or2392 fits','not executed','NOT_REQUESTED; no training','prior method audit')]
 for f,p,a,e in additions:rows.append({'analysis_family':f,'previous_status':p,'v38_action':a,'evidence':e,'boundary':CLAIM})
 csv('coverage_matrix.csv',pd.DataFrame(rows))

def main():
 coverage();d,oof=integrity_and_load();overall=analysis(d,oof);progress('FIVE_FAMILY_DIAGNOSTICS_PASS',overall[['track','model','loss','coverage90']].to_dict('records'));replay(oof);portfolio(d);controls();csv('source_ablation_reuse.csv',pd.read_csv(R/'reports/model_risk_v3/source_ablation.csv'));progress('ANALYTICS_COMPLETE',{'all_rows':len(d),'new_fits':0,'new_calibrators':0,'remaining':'tests, independent validation, technical report and immutable seal'})
if __name__=='__main__':main()
