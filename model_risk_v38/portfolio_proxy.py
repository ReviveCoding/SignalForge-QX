"""Fixed historical P0 signal mappings only: no executable orders/economic returns."""
from .source import *
COMMODITIES={'GLD','SLV','UNG','USO'}
def weights(assets,scores,mapping):
 assets=np.asarray(assets);scores=np.asarray(scores,float)
 if not np.isfinite(scores).all() or len(set(assets))!=len(assets):raise ValueError('Invalid signal input')
 order=sorted(range(len(assets)),key=lambda i:(-scores[i],assets[i]));w=np.zeros(len(assets))
 if mapping in ['top1','top2']:w[order[:int(mapping[-1])]]=.25
 elif mapping=='linear_rank':w[order]=np.arange(len(assets),0,-1);w/=w.sum();w=np.minimum(w,.25)
 else:raise ValueError('Unknown fixed mapping')
 c=np.array([a in COMMODITIES for a in assets]);mass=w[c].sum()
 if mass>.35:w[c]*=.35/mass
 return w

def validate_weights(assets,w):
 w=np.asarray(w,float);c=np.array([a in COMMODITIES for a in assets])
 if not np.isfinite(w).all() or (w<0).any() or (w>.25+1e-12).any() or w.sum()>1+1e-12 or w[c].sum()>.35+1e-12:raise ValueError('Weight caps')
 return True

def portfolio(d):
 previous=pd.read_csv(R/'reports/model_risk_v3/signal_weights.csv');previous.decision_time=aware(previous.decision_time)
 saved=previous.set_index(['track','model','decision_time','seed','mapping','asset']).weight
 records=[];generated=[]
 for (track,model,seed),g in d.groupby(['track','model','seed']):
  last={}
  for date,h in g.groupby('decision_time',sort=True):
   h=h.sort_values('asset');assets=h.asset.to_numpy();control=d[(d.track==track)&(d.model=='strong_I0')&(d.seed==seed)&(d.decision_time==date)].sort_values('asset')
   for mapping in ['top1','top2','linear_rank']:
    w=weights(assets,h.q50,mapping);b=weights(assets,control.q50,mapping);validate_weights(assets,w)
    for asset,value in zip(assets,w):
     key=(track,model,date,seed,mapping,asset)
     if model!='SIA_calibrated_v34' and abs(saved.loc[key]-value)>1e-12:raise ValueError('Historical P0 mapping differs')
     generated.append(dict(zip(['track','model','decision_time','seed','mapping','asset'],key),weight=value))
    gross=float(w.sum());hhi=float(np.sum((w/gross)**2)) if gross else 0;turn=np.nan if mapping not in last else float(np.abs(w-last[mapping]).sum());last[mapping]=w
    records.append({'track':track,'model':model,'decision_time':date,'seed':seed,'mapping':mapping,'gross_signal_weight':gross,'cash_remainder':1-gross,'HHI_gross_normalized':hhi,'effective_holdings':1/hhi if hhi else 0,'raw_HHI':float(np.sum(w**2)),'max_asset_weight':float(w.max()),'commodity_weight':float(sum(v for a,v in zip(assets,w) if a in COMMODITIES)),'L1_vs_strong':float(np.abs(w-b).sum()),'L1_turnover_proxy':turn,'top_exposure_asset':str(assets[np.argmax(w)]),'scope':'P0_HYPOTHETICAL_SIGNAL_ONLY_NOT_PNL','exante_high_vol_assets':int((h.realized_vol_20_regime=='HIGH').sum())})
 csv('P0_portfolio_materiality_proxy.csv',pd.DataFrame(records));csv('P0_signal_weights.csv',pd.DataFrame(generated));progress('P0_PROXY_PASS',{'groups':len(records),'old_weights_verified':len(saved),'calibrated_mapping_extension':'same immutable fixed rule','PnL':False})

def controls():
 m=readj(R/'reports/model_risk_v3/monitoring.json');risk=pd.read_csv(R/'reports/model_risk_v3/risk_register.csv');records=[]
 replay=pd.read_csv(OUT/'monitor_summary.csv');breaches=m['retrospective_reference_breaches'];actual=readj(OUT/'panel_integrity.json')
 for item in m['catalog']:
  indicator=item['indicator'];s=indicator.lower();norm=item['source']=='NORMATIVE_SCIENTIFIC_GUARD';evidence='reports/model_risk_v3/monitoring.json (historical normative/candidate definition)';status=item['status'];blocker='No operational freeze or appointed independent owner';measure='Historical saved-panel audit only';observed='NOT_MEASURED_WITH_AUTHENTIC_SOURCE_CLOCK'
  if any(k in s for k in ['nonfinite','crossing','duplicate','hash','checkpoint']):
   evidence='panel_integrity.json; independent_validation.json; TEST_ONLY fail-closed guards';measure='finite ordered unique forecasts and immutable bundle hashes';observed='0 saved-panel violations';status='MEASURED_INTEGRITY_GUARD_NOT_DEPLOYED'
  elif 'error' in s or 'coverage' in s:
   evidence='monitor_summary.csv; calibration_by_slice.csv';measure='maturity-qualified SIA q95 rolling monitor; outer reliability descriptive';observed='see threshold/date-specific dry-run exceedances';status='SIA_OOF_REPLAY_IMPLEMENTED; BAR_OOF_BLOCKED'
  elif 'age' in s or 'availability' in s or 'source' in s:
   evidence='ex_ante_features.csv historical; monitor_clock_validation.json';measure='pre-outer train-only age candidates; historical Tier B dependency checks';status='TIER_B_ONLY; OOF_SOURCE_CLOCK_MISSING'
  elif 'seed' in s or 'disagreement' in s:
   evidence='seed_functional_stability.csv; challenger_disagreement.csv';measure='date/asset seed spans, sign/rank disagreement';observed='descriptive numeric rows; no controlled alarm probability'
  elif 'turnover' in s or 'concentration' in s:
   evidence='P0_portfolio_materiality_proxy.csv';measure='fixed capped signal exposure HHI/L1; no trading';observed='no cap violations';status='P0_PROXY_NOT_P1'
  records.append({'kind':'MONITOR','id':indicator,'evidence':evidence,'measurable_control':measure,'observed_breach':observed,'source_clock':'Tier B reconstructed, not Tier A','cadence':item['cadence'],'owner':'PROPOSED_NOT_APPOINTED: '+item['owner'],'escalation':item['action'],'control_type':'HARD_INTEGRITY_STOP_NORMATIVE' if norm else 'ADVISORY_UNFROZEN','status':status,'blocker':blocker,'deployment':False})
 for _,item in risk.iterrows():
  blocker='independent cohort / authentic PIT / P1 / operational freeze absent'
  if item.risk_id=='R09':blocker='SIA OOF available (441/66 dates); BAR OOF absent; 26/52-lookback monitoring remains support constrained'
  records.append({'kind':'RISK','id':item.risk_id,'evidence':item.evidence_or_risk+'; new source_index.json','measurable_control':item.mitigation,'observed_breach':item.failure_mode,'source_clock':'Tier B reconstructed','cadence':'research batch review (proposed)','owner':'PROPOSED_NOT_APPOINTED: '+item.proposed_owner,'escalation':item.action,'control_type':'ADVISORY_RESEARCH_QUARANTINE','status':'NO_REGULATORY_OR_EXTERNAL_VALIDATION_CLAIM','blocker':blocker,'deployment':False})
 if len(m['catalog'])!=21 or len(risk)!=11:raise ValueError('Control inventory changed')
 csv('model_risk_control_map.csv',pd.DataFrame(records));writej('candidate_disposition.json',{'strong_I0':'RETAIN_FROZEN_REFERENCE_NOT_PROMOTED','SIA_v2':'RESEARCH_ONLY','BAR':'QUARANTINED_NEGATIVE_PAIRED_RESULTS','SIA_calibrated_v34':'QUARANTINED_CALIBRATION_DAMAGE','ABC':'SYNTHETIC_PROTOTYPE_ONLY','no_external_validator':True,'Tier_A':0,'P1':'0/27088 qualified','reserved_final':'SEALED','future_independent_dates':0})
