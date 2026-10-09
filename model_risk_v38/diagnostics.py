"""Read-only numerical diagnostics; date clusters, fixed slices, no fitted parameters."""
from .source import *
from scipy.stats import spearmanr,kendalltau
PAIRS=[('SIA_v2','strong_I0'),('SIA_calibrated_v34','SIA_v2'),('BAR','strong_I0')]
def enrich(df):
 d=df.copy();y=d.target.to_numpy()[:,None];q=d[QC].to_numpy();e=y-q
 raw=np.maximum(Q*e,(Q-1)*e);norm=raw/d.scale.to_numpy()[:,None]
 for j,c in enumerate(QC):d['raw_pinball_'+c]=raw[:,j];d['pinball_'+c]=norm[:,j];d['coverage_'+c]=(y[:,0]<=q[:,j]).astype(float)
 d['loss']=norm.mean(1);d['raw_loss']=raw.mean(1);d['median_residual']=d.target-d.q50;d['normalized_bias']=d.median_residual/d.scale
 d['abs_error']=d.median_residual.abs();d['normalized_abs_error']=d.abs_error/d.scale
 for level,lo,hi,alpha in [(80,'q10','q90',.2),(90,'q05','q95',.1)]:
  d['width'+str(level)]=d[hi]-d[lo];d['normalized_width'+str(level)]=d['width'+str(level)]/d.scale
  d['coverage'+str(level)]=((d.target>=d[lo])&(d.target<=d[hi])).astype(float)
  d['interval_score'+str(level)]=d['width'+str(level)]+2/alpha*(d[lo]-d.target).clip(lower=0)+2/alpha*(d.target-d[hi]).clip(lower=0)
  d['normalized_interval_score'+str(level)]=d['interval_score'+str(level)]/d.scale
 d['max_abs_forecast']=np.abs(q).max(1);d['catastrophic']=(d.max_abs_forecast>100).astype(int)
 return d

def date_series(g,columns):return g.groupby(['decision_time','asset'])[columns].mean().groupby('decision_time').mean().sort_index()
def mean_equal(g,col):return float(date_series(g,[col])[col].mean())
def calibration(g):
 out={'rows':len(g),'distinct_dates':g.decision_time.nunique(),'assets':g.asset.nunique(),'seeds':g.seed.nunique(),'status':'DESCRIPTIVE' if g.decision_time.nunique()>=13 else 'UNDERPOWERED'}
 cols=['loss','raw_loss','normalized_bias','normalized_abs_error','coverage80','coverage90','width80','width90','normalized_width80','normalized_width90','interval_score80','interval_score90','normalized_interval_score80','normalized_interval_score90']
 for c in cols:out[c]=mean_equal(g,c)
 dev=[]
 for tau,c in zip(Q,QC):
  cov=mean_equal(g,'coverage_'+c);out['coverage_'+c]=cov;out['coverage_pp_'+c]=100*(cov-tau);dev.append(abs(cov-tau))
  out['pinball_'+c]=mean_equal(g,'pinball_'+c);out['raw_pinball_'+c]=mean_equal(g,'raw_pinball_'+c);out['expected_tail_dates_'+c]=out['distinct_dates']*min(tau,1-tau)
 out['descriptive_mean_abs_quantile_coverage_deviation']=float(np.mean(dev));out['max_abs_forecast']=float(g.max_abs_forecast.max());out['p99_abs_forecast']=float(np.quantile(np.abs(g[QC].to_numpy()),.99));out['catastrophic_rows']=int(g.catastrophic.sum());out['crossings']=int((np.diff(g[QC].to_numpy(),axis=1)<0).sum());out['tail_support']='EXPECTED_TAIL_DATES_BELOW_20' if out['distinct_dates']*.05<20 else 'DESCRIPTIVE_ONLY'
 return out

def slices(g):
 yield 'ALL','ALL',g
 for dim in readj(OUT/'frozen_plan.json')['slices']:
  if dim in g:
   for val,part in g.groupby(dim,dropna=False):yield dim,str(val),part

def paired(d,candidate,control):
 a=d[d.model==candidate].set_index(KEY).sort_index();b=d[d.model==control].set_index(KEY).sort_index()
 if not a.index.equals(b.index):raise ValueError('Pair mismatch')
 x=a.reset_index();x['control_loss']=b.loss.to_numpy();x['delta']=a.loss.to_numpy()-b.loss.to_numpy();x['control_q50']=b.q50.to_numpy();x['candidate']=candidate;x['control']=control
 for c in QC:x['delta_'+c]=a['pinball_'+c].to_numpy()-b['pinball_'+c].to_numpy()
 return x

def block_bounds(values,block,draws=2000,seed=38001):
 x=np.asarray(values,float);n=len(x)
 if n<block:raise ValueError('Insufficient date clusters')
 rng=np.random.default_rng(seed+block);starts=rng.integers(0,n,(draws,int(np.ceil(n/block))));ix=(starts[:,:,None]+np.arange(block))%n;means=x[ix.reshape(draws,-1)[:,:n]].mean(1)
 return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def analysis(d,oof):
 rows=[]
 for (t,m),g in d.groupby(['track','model']):
  for dim,val,part in slices(g):rows.append({'track':t,'model':m,'slice':dim,'value':val,**calibration(part)})
 csv('calibration_by_slice.csv',pd.DataFrame(rows));overall=pd.DataFrame([x for x in rows if x['slice']=='ALL']);csv('overall_metrics.csv',overall)
 gap=[]
 for t,g in oof.groupby('track'):
  metrics=calibration(enrich(g));outer=overall[(overall.track==t)&(overall.model=='SIA_v2')].iloc[0]
  for c in ['loss','coverage_q05','coverage_q10','coverage_q50','coverage_q90','coverage_q95','coverage90','normalized_abs_error']:
   gap.append({'track':t,'metric':c,'OOF':metrics[c],'outer':outer[c],'outer_minus_OOF':outer[c]-metrics[c],'OOF_dates':metrics['distinct_dates'],'outer_dates':52,'claim':'DIFFERENT_CHRONOLOGICAL_COHORTS_NOT_CAUSAL'})
 csv('OOF_outer_gap.csv',pd.DataFrame(gap))
 attribution=[];decomp=[];boot=[];influence=[];challenger=[];regime=[];concentration=[]
 for t,g in d.groupby('track'):
  strong=date_series(g[g.model=='strong_I0'],['loss']).loss;stressdates=strong.sort_values(ascending=False).index
  for m,h in g.groupby('model'):
   losses=date_series(h,['loss']).loss;assets=h.groupby('asset').loss.mean();concentration.append({'track':t,'model':m,'USO_loss_share':float(assets.get('USO',0)/assets.sum()),'top5_week_loss_share':float(losses.nlargest(5).sum()/losses.sum()),'dates':52})
  for ca,co in PAIRS:
   x=paired(g,ca,co);overall_delta=mean_equal(x,'delta');base=mean_equal(x,'control_loss');series=date_series(x,['delta']).delta
   for block in [4,8,13]:
    lo,hi=block_bounds(series.to_numpy(),block);boot.append({'track':t,'candidate':ca,'control':co,'delta':overall_delta,'relative_deterioration_pct':100*overall_delta/base,'block':block,'draws':2000,'dates':52,'lower_descriptive':lo,'upper_descriptive':hi,'claim':CLAIM})
   for dim,val,part in slices(x):
    contribution=float(part.delta.sum()/len(x));attribution.append({'track':t,'candidate':ca,'control':co,'dimension':dim,'value':val,'rows':len(part),'distinct_dates':part.decision_time.nunique(),'signed_loss_budget_contribution':contribution,'within_slice_date_equal_delta':mean_equal(part,'delta'),'overall_delta':overall_delta,'support':'DESCRIPTIVE' if part.decision_time.nunique()>=13 else 'UNDERPOWERED'})
    if dim!='ALL':regime.append({'track':t,'candidate':ca,'control':co,'slice':dim,'value':val,'dates':part.decision_time.nunique(),'delta':mean_equal(part,'delta'),'corrects_control_loss_fraction':float((part.delta<0).mean()),'status':'DESCRIPTIVE' if part.decision_time.nunique()>=13 else 'UNDERPOWERED'})
    if ca=='SIA_calibrated_v34':
     for c in QC:decomp.append({'track':t,'slice':dim,'value':val,'quantile':c,'rows':len(part),'dates':part.decision_time.nunique(),'delta':mean_equal(part,'delta_'+c),'signed_total_loss_contribution':float(part['delta_'+c].sum()/len(x)/5)})
   for c in QC:attribution.append({'track':t,'candidate':ca,'control':co,'dimension':'quantile','value':c,'rows':len(x),'distinct_dates':52,'signed_loss_budget_contribution':float(x['delta_'+c].mean()/5),'within_slice_date_equal_delta':float(x['delta_'+c].mean()),'overall_delta':overall_delta,'support':'DESCRIPTIVE'})
   for asset in sorted(x.asset.unique()):influence.append({'track':t,'candidate':ca,'control':co,'operation':'LEAVE_ONE_ASSET_OUT','excluded':asset,'original_delta':overall_delta,'counterfactual_delta':mean_equal(x[x.asset!=asset],'delta'),'primary_rows_retained':len(x),'claim':'COUNTERFACTUAL_INFLUENCE_ONLY'})
   for date in stressdates[:5]:influence.append({'track':t,'candidate':ca,'control':co,'operation':'LEAVE_ONE_STRONG_BASELINE_STRESS_WEEK_OUT','excluded':str(date),'original_delta':overall_delta,'counterfactual_delta':mean_equal(x[x.decision_time!=date],'delta'),'primary_rows_retained':len(x),'claim':'COUNTERFACTUAL_INFLUENCE_ONLY'})
   for k in [1,3,5]:influence.append({'track':t,'candidate':ca,'control':co,'operation':'REMOVE_TOP_K_STRONG_BASELINE_STRESS_WEEKS','excluded':str(k),'original_delta':overall_delta,'counterfactual_delta':mean_equal(x[~x.decision_time.isin(stressdates[:k])],'delta'),'primary_rows_retained':len(x),'claim':'COUNTERFACTUAL_INFLUENCE_ONLY'})
   for (date,seed),part in x.groupby(['decision_time','seed']):
    a=part.sort_values('asset');v=a.q50.to_numpy();b=a.control_q50.to_numpy();orderA=sorted(range(8),key=lambda i:(-v[i],a.asset.iloc[i]));orderB=sorted(range(8),key=lambda i:(-b[i],a.asset.iloc[i]));truth=np.sign(a.target.to_numpy());sa=set(orderA[:2]);sb=set(orderB[:2])
    challenger.append({'track':t,'candidate':ca,'control':co,'decision_time':date,'seed':seed,'spearman':float(spearmanr(v,b).statistic) if np.ptp(v)>0 and np.ptp(b)>0 else np.nan,'kendall':float(kendalltau(v,b).statistic) if np.ptp(v)>0 and np.ptp(b)>0 else np.nan,'top1_switch':orderA[0]!=orderB[0],'top2_jaccard':len(sa&sb)/len(sa|sb),'candidate_ties':8-len(np.unique(v)),'control_ties':8-len(np.unique(b)),'median_sign_disagreement':float(np.mean(np.sign(v)!=np.sign(b))),'candidate_sign_truth':float(np.mean(np.sign(v)==truth)),'control_sign_truth':float(np.mean(np.sign(b)==truth)),'near_zero_control_fraction':float(np.mean(np.abs(b/a.scale.to_numpy())<1e-3)),'mean_delta':float(a.delta.mean()),'corrected_loss_assets':int((a.delta<0).sum()),'hurt_loss_assets':int((a.delta>0).sum())})
 csv('paired_loss_attribution.csv',pd.DataFrame(attribution));csv('calibration_delta_decomposition.csv',pd.DataFrame(decomp));csv('regime_challenger.csv',pd.DataFrame(regime));csv('challenger_disagreement.csv',pd.DataFrame(challenger));csv('stress_influence.csv',pd.DataFrame(influence));csv('paired_date_block_bounds.csv',pd.DataFrame(boot));csv('loss_concentration.csv',pd.DataFrame(concentration))
 residual=[];rolling=[];stability=[];hetero=[]
 for (t,m),g in d.groupby(['track','model']):
  for dim,val,h in [('ALL','ALL',g)]+[('asset',a,p) for a,p in g.groupby('asset')]:
   s=date_series(h,['normalized_bias','normalized_abs_error','loss','coverage90'])
   for metric in ['normalized_bias','normalized_abs_error','loss']:
    for lag in [1,2,3,4,8]:residual.append({'track':t,'model':m,'slice':dim,'value':val,'metric':metric,'lag':lag,'correlation':s[metric].autocorr(lag),'dates':len(s),'pairs':len(s)-lag,'claim':'DESCRIPTIVE_NOT_INDEPENDENT_ERRORS'})
   for window in [13,26]:
    for date,row in s.rolling(window,min_periods=window).mean().dropna().iterrows():rolling.append({'track':t,'model':m,'slice':dim,'value':val,'decision_time':date,'window':window,**row.to_dict()})
  for (date,asset),h in g.groupby(['decision_time','asset']):
   preds=h[QC].to_numpy();scale=float(h.scale.iloc[0]);width=h.width90.to_numpy();stability.append({'track':t,'model':m,'decision_time':date,'asset':asset,'median_span_normalized':float(np.ptp(preds[:,2])/scale),'median_sign_flip':len(np.unique(np.sign(preds[:,2])))>1,'width90_seed_std_normalized':float(np.std(width)/scale),'max_quantile_span_normalized':float(np.ptp(preds,axis=0).max()/scale),'deterministic_baseline_duplicates':m=='strong_I0','realized_vol_20_regime':h.realized_vol_20_regime.iloc[0],'any_source_OOD':h.any_source_OOD.iloc[0]})
  for col in ['realized_vol_20','return_1w','return_4w']:
   z=g.groupby(['decision_time','asset'])[[col,'normalized_abs_error','normalized_bias']].mean();rho=spearmanr(z[col],z.normalized_abs_error,nan_policy='omit').statistic;hetero.append({'track':t,'model':m,'exante_feature':col,'rank_association_abs_error':float(rho),'dates':52,'claim':'ASSET_DATE_ASSOCIATION_NOT_CAUSAL_OR_INDEPENDENT'})
 csv('residual_autocorrelation.csv',pd.DataFrame(residual));csv('residual_rolling.csv',pd.DataFrame(rolling));csv('seed_functional_stability.csv',pd.DataFrame(stability));csv('heteroscedasticity_associations.csv',pd.DataFrame(hetero))
 return overall
