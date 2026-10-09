"""Post-hoc diagnostics only: no fitting, trading, or production threshold selection."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau
Q=np.array([.05,.10,.50,.90,.95]);COLS=['q05','q10','q50','q90','q95']
MODELS=['original_RGMF','strong_I0','SIA_v2','BAR']
COMMODITIES={'GLD','SLV','UNG','USO'}

def check_grid(f,dates=52,assets=8,seeds=(11,37,71)):
    required={'decision_time','asset','seed','target','scale',*COLS}
    if not required<=set(f):raise ValueError('Forecast schema missing')
    times=pd.to_datetime(f.decision_time,utc=True)
    if (times>=pd.Timestamp('2024-01-01T00:00Z')).any():raise PermissionError('Reserved cohort forbidden')
    if f.duplicated(['decision_time','asset','seed']).any():raise ValueError('Duplicate identities')
    if times.nunique()!=dates or f.asset.nunique()!=assets or set(f.seed)!=set(seeds) or len(f)!=dates*assets*len(seeds):raise ValueError('Incomplete exact evaluation grid')
    for _,g in f.groupby('decision_time'):
        if len(g)!=assets*len(seeds) or g.groupby('seed').asset.nunique().min()!=assets:raise ValueError('Partial date')
    a=f[['target','scale',*COLS]].to_numpy(dtype=float)
    if not np.isfinite(a).all() or (f.scale<=0).any():raise ValueError('Invalid forecasts/target/scale')
    if (np.diff(f[COLS],axis=1)<0).any():raise ValueError('Quantile crossing')
    for _,g in f.groupby(['decision_time','asset']):
        if g.target.nunique()!=1 or g.scale.nunique()!=1:raise ValueError('Seed target or scale mismatch')
    return True

def augment(f):
    f=f.copy();q=f[COLS].to_numpy();y=f.target.to_numpy();s=f.scale.to_numpy();e=y[:,None]-q;pin=np.maximum(Q*e,(Q-1)*e)
    for j in range(5):f['coverage_'+COLS[j]]=(y<=q[:,j]).astype(float);f['pinball_'+COLS[j]]=pin[:,j]/s;f['raw_pinball_'+COLS[j]]=pin[:,j]
    f['loss']=pin.mean(1)/s;f['raw_loss']=pin.mean(1);f['median_residual']=y-q[:,2];f['abs_error']=np.abs(f.median_residual);f['normalized_abs_error']=f.abs_error/s
    for a,lo,hi in [(80,1,3),(90,0,4)]:
        alpha=1-a/100;l=q[:,lo];u=q[:,hi];f[f'width{a}']=u-l;f[f'normalized_width{a}']=(u-l)/s;f[f'coverage{a}']=((y>=l)&(y<=u)).astype(float);f[f'interval_score{a}']=((u-l)+2/alpha*np.maximum(l-y,0)+2/alpha*np.maximum(y-u,0))/s
    f['max_abs_forecast']=np.abs(q).max(1);f['catastrophic']=(f.max_abs_forecast>100).astype(int)
    return f

def bootstrap_dates(f,columns,blocks=(4,8,13),draws=2000,seed=20261008):
    g=f.groupby('decision_time')[columns].mean().sort_index();v=g.to_numpy();out={}
    for b in blocks:
        if len(v)<b:out[str(b)]={'state':'UNDERPOWERED','dates':len(v)};continue
        rng=np.random.default_rng(seed+b);starts=rng.integers(0,len(v)-b+1,size=(draws,int(np.ceil(len(v)/b))));idx=(starts[:,:,None]+np.arange(b)).reshape(draws,-1)[:,:len(v)];boot=v[idx].mean(1);bounds=np.quantile(boot,[.025,.975],axis=0)
        out[str(b)]={'independent_dates':len(v),'descriptive_only':True,'columns':{c:{'mean':float(v[:,j].mean()),'interval95':[float(bounds[0,j]),float(bounds[1,j])]} for j,c in enumerate(columns)}}
    return out

def summary(f,ci=False):
    dates=f.decision_time.nunique();coverage=[float(f['coverage_'+c].mean()) for c in COLS];bydate=f.groupby('decision_time').loss.mean().sort_values(ascending=False);total=bydate.sum();asset_loss=f.groupby('asset').loss.sum()
    s={'rows':len(f),'distinct_dates':int(dates),'assets':int(f.asset.nunique()),'seeds':int(f.seed.nunique()),'support':'UNDERPOWERED' if dates<13 else 'DESCRIPTIVE_DEVELOPMENT_ONLY','normalized_pinball':float(f.groupby('decision_time').loss.mean().mean()),'raw_pinball':float(f.raw_loss.mean()),'coverage':coverage,'coverage_deviation_pp':(100*(np.array(coverage)-Q)).tolist(),'pinball_by_quantile':[float(f['pinball_'+c].mean()) for c in COLS],'raw_pinball_by_quantile':[float(f['raw_pinball_'+c].mean()) for c in COLS],'signed_median_residual':float(f.median_residual.mean()),'median_residual_median':float(f.median_residual.median()),'MAE':float(f.abs_error.mean()),'MedAE':float(f.abs_error.median()),'normalized_MAE':float(f.normalized_abs_error.mean()),'error_p95':float(f.abs_error.quantile(.95)),'error_p99':float(f.abs_error.quantile(.99)),'top5_date_abs_error_fraction':float(f.groupby('decision_time').abs_error.mean().nlargest(5).sum()/f.groupby('decision_time').abs_error.mean().sum()) if f.abs_error.sum()>0 else 0.,'loss_p95':float(f.loss.quantile(.95)),'loss_p99':float(f.loss.quantile(.99)),'max_abs_forecast':float(f.max_abs_forecast.max()),'p99_abs_forecast':float(f.max_abs_forecast.quantile(.99)),'catastrophic_rows':int(f.catastrophic.sum()),'crossings':int((np.diff(f[COLS],axis=1)<0).sum()),'top5_date_loss_fraction':float(bydate.head(5).sum()/total),'asset_loss_fractions':(asset_loss/asset_loss.sum()).to_dict()}
    for a in (80,90):s.update({f'coverage{a}':float(f[f'coverage{a}'].mean()),f'width{a}':float(f[f'width{a}'].mean()),f'normalized_width{a}':float(f[f'normalized_width{a}'].mean()),f'normalized_interval_score{a}':float(f[f'interval_score{a}'].mean())})
    if ci:s['date_block_uncertainty']=bootstrap_dates(f,['loss',*['coverage_'+c for c in COLS],'coverage80','coverage90','normalized_width90','normalized_abs_error'])
    return s

def train_only_threshold(values,dates,available,cutoff,*,cohort,quantiles=(.95,.99),minimum_dates=52):
    if cohort!='MATURE_PRE_OUTER':raise PermissionError('No threshold selection on outer outcomes')
    d=pd.to_datetime(dates,utc=True);a=pd.to_datetime(available,utc=True);c=pd.Timestamp(cutoff)
    if c.tzinfo is None or (d>=pd.Timestamp('2024-01-01T00:00Z')).any() or (d>=c).any() or (a>c).any():raise PermissionError('Maturity/cutoff breach')
    v=np.asarray(values,dtype=float);good=np.isfinite(v)
    if len(v)!=len(d) or not good.any():raise ValueError('Threshold cohort alignment')
    support=pd.DatetimeIndex(d[good]).nunique()
    if support<minimum_dates:return {'state':'UNDERPOWERED','distinct_dates':int(support),'thresholds':None}
    return {'state':'TRAIN_ONLY_CANDIDATE_NOT_OPERATIONALLY_FROZEN','distinct_dates':int(support),'rows':int(good.sum()),'cutoff':c.isoformat(),'thresholds':{str(q):float(np.quantile(v[good],q)) for q in quantiles},'false_positive_rate_controlled':False,'outer_outcomes_used':False}

def tercile(v,bounds):
    v=np.asarray(v,dtype=float);out=np.full(len(v),'MISSING',dtype=object);ok=np.isfinite(v);out[ok]=np.where(v[ok]<=bounds[0],'LOW',np.where(v[ok]<=bounds[1],'MID','HIGH'));return out

def rank_compare(a,b):
    a=a.sort_values('asset');b=b.sort_values('asset');assert a.asset.tolist()==b.asset.tolist()
    ar=a.sort_values(['q50','asset'],ascending=[False,True]).asset.tolist();br=b.sort_values(['q50','asset'],ascending=[False,True]).asset.tolist();x=a.q50.to_numpy();y=b.q50.to_numpy()
    return {'spearman':None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic),'kendall':None if np.ptp(x)==0 or np.ptp(y)==0 else float(kendalltau(x,y).statistic),'ties_a':int(len(x)-len(np.unique(x))),'ties_b':int(len(y)-len(np.unique(y))),'top1_switch':ar[0]!=br[0],'top2_jaccard':len(set(ar[:2])&set(br[:2]))/len(set(ar[:2])|set(br[:2])),'top1_a':ar[0],'top1_b':br[0]}

def capped_weights(assets,scores,mapping):
    assets=np.asarray(assets);scores=np.asarray(scores);order=sorted(range(len(assets)),key=lambda i:(-scores[i],assets[i]));w=np.zeros(len(assets))
    if mapping in ('top1','top2'):w[order[:int(mapping[-1])]]=.25
    elif mapping=='linear_rank':
        w[order]=np.arange(len(assets),0,-1);w/=w.sum();w=np.minimum(w,.25)
    else:raise ValueError('Unknown fixed weight mapping')
    commodity=np.array([a in COMMODITIES for a in assets]);mass=w[commodity].sum()
    if mass>.35:w[commodity]*=.35/mass
    if (w<0).any() or (w>.25+1e-12).any() or w.sum()>1+1e-12 or w[commodity].sum()>.35+1e-12:raise ValueError('Signal weight constraint')
    return w

def economic_claim(*,p1_qualified=False):
    if not p1_qualified:raise PermissionError('BLOCKED_P1: signal proxies are not PnL')
    raise PermissionError('This diagnostics package never computes economic performance')

def model_error_threshold(family,has_mature_model_OOF):
    if family in ('SIA_v2','BAR','original_RGMF') and not has_mature_model_OOF:return {'state':'BLOCKED_OOF_UNAVAILABLE','calibrator_fit':False,'thresholds':None}
    return {'state':'DIAGNOSTICS_ONLY_NO_CALIBRATOR_FIT'}
def canonical_keys(pairs):
    result=[]
    for d,a in pairs:
        t=pd.Timestamp(d)
        if t.tzinfo is None:raise ValueError('Aware decision key required')
        result.append((t.tz_convert('UTC').isoformat(),str(a)))
    return result

def replay_status(cpu,saved,atol=2e-7,rtol=2e-6):
    cpu=np.asarray(cpu);saved=np.asarray(saved)
    if cpu.shape!=saved.shape or not np.isfinite(cpu).all() or not np.isfinite(saved).all():raise ValueError('Corrupt replay')
    passed=bool(np.allclose(cpu,saved,atol=atol,rtol=rtol))
    return {'strict_replay_passed':passed,'state':'PASSED_WITH_DECLARED_TOLERANCE' if passed else 'CPU_BACKEND_DIAGNOSTIC_NOT_EXACT_CUDA_REPLAY','max_abs_difference':float(np.abs(cpu-saved).max()),'atol':atol,'rtol':rtol,'primary_metrics_use_saved_CUDA_only':True}

def panel_bundle_metadata(code_digest):
    if not isinstance(code_digest,str) or len(code_digest)!=64:raise ValueError('Diagnostic code digest required')
    return {'diagnostics_hash':code_digest,'reserved_access':False,'training':False,'scope':'saved CUDA panel metrics; CPU perturbations separately limited'}
