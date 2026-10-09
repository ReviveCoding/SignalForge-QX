"""Execute CPU-only immutable prediction audits; never calls fit or CUDA."""
import os,sys,json,io,itertools,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from signalforge.runtime import code_hash,digest,file_hash,atomic_json,validate_bundle,commit_bundle,now
from signalforge.successor import ledger_snapshot
from signalforge.panels import track_folds
from build_v33_readiness import load_original,ORIGINAL,ORIGINAL_RUNTIME,bytes_hash
from studies_v3.runner import intact,external_hash
from studies_v3.bar_model import BAR,source_tensors
from diagnostics_v3.core import *
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering');O=R/'reports/model_risk_v3'
FROZEN='471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,Path)):return str(x)
    return x

def save(name,x):atomic_json(O/(name+'.json'),clean(x))
def event(stage,**kw):
    v={'created_at':now(),'stage':stage,'GPU_training_fits':0,'reserved_access':False,**kw};save('live_progress',v);print(json.dumps(clean(v)),flush=True)
def diag_hash():return digest({str(p.relative_to(R)):file_hash(p) for p in sorted((R/'diagnostics_v3').glob('*')) if p.suffix in {'.py','.ps1'}})
def preserve():
    v=intact();assert code_hash(R)==FROZEN
    for receipt in ['v33_minimal_completion_v2','bar_completion_v1','bar_handoff_evidence_index_v1']:
        c=load(R/'reports'/f'{receipt}.json');validate_bundle(T/c['relative_bundle'])
        for p,h in c['files'].items():assert file_hash(R/p)==h,p
    old=load(R/'reports/v33_minimal_results_v2.json');cfg=load(R/'configs/v33_minimal_execution_v2r1.json')
    for key,field,cap in [('pilot','pilot_ledger',300),('sia','study_ledger',1800)]:assert ledger_snapshot(cfg['ledgers'][key],cap)==old[field]
    for p,h in load(R/'reports/v33_isolation_receipt.json')['copied_files'].items():assert file_hash(ORIGINAL/p.replace('\\','/'))==h
    # Read only observation databases: no recovery, metering, or charge calls.
    import sqlite3
    val=load(R/'reports/bar_independent_validation_v1.json')
    for name,key in [('study','study_meter_records'),('integration','integration_meter_records'),('validation','independent_reload_meter_records')]:
        path=T/'ledger'/f'v33_budget_free_v3_bar_{name}_v1.sqlite'
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row;rows=[dict(r) for r in db.execute('SELECT * FROM observations ORDER BY rowid')]
        assert rows==val[key]
    return v

def tests():
    p=subprocess.run([sys.executable,'-m','pytest','diagnostics_v3/test_model_risk.py','-q','-p','no:cacheprovider'],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(R)+':'+str(R/'src')));print(p.stdout+p.stderr,flush=True);save('tests',{'exit_code':p.returncode,'log':p.stdout+p.stderr,'diagnostics_hash':diag_hash(),'frozen_source_hash':code_hash(R)});assert p.returncode==0

def load_forecasts():
    v2=load(R/'configs/v33_minimal_execution_v2r1.json');bp=load(R/'reports/bar_frozen_plan_v1.json');sia=load(R/'reports/v33_minimal_study_v2.json')['results'];bar=load(R/'reports/bar_fits_v1.json')['results'];panel=[];folds={};receipts={}
    for track,year in [('Main-A',2020),('Nested-B',2022)]:
        key=f'{track}:{year}';vb=T/v2['data_bundles'][key];bb=T/bp['data_bundles'][key]
        for b in [vb,bb]:validate_bundle(b);receipts[str(b/'receipt.json')]=file_hash(b/'receipt.json')
        vz=np.load(vb/'arrays.npz');bz=np.load(bb/'arrays.npz');vm=load(vb/'data.json');bm=load(bb/'metadata.json')
        assert not vm['control_missing'];assert np.array_equal(vz['target'],bz['target']) and np.array_equal(vz['scale'],bz['scale'])
        assert canonical_keys(vm['keys'])==canonical_keys(bm['date_asset_keys'])
        for k in ['fold_id','source_id','normalizer_id']:assert vm[k]==bm[k]
        assert vm['target_contract_id']==bm['target_id']==bytes_hash(vz['target'])
        base=pd.DataFrame(bm['date_asset_keys'],columns=['decision_time','asset']);base['target']=vz['target'];base['scale']=vz['scale'];base['track']=track;base['year']=year;base['fold_id']=bm['fold_id'];base['source_id']=bm['source_id'];base['normalizer_id']=bm['normalizer_id'];base['target_id']=bm['target_id']
        for label,controls in vm['controls'].items():
            for row in controls:
                b=ORIGINAL_RUNTIME/'artifacts/track_development'/row['run_id'];validate_bundle(b);assert file_hash(b/'receipt.json')==row['receipt_sha256'];receipts[str(b/'receipt.json')]=row['receipt_sha256'];assert np.array_equal(np.load(b/'predictions.npz')['quantiles'],vz[label+'_'+str(row['seed'])])
        for seed in [11,37,71]:
            sf=next(f for f in sia if (f['track'],f['year'],f['seed'])==(track,year,seed));bf=next(f for f in bar if (f['track'],f['year'],f['seed'])==(track,year,seed))
            ss=T/sf['relative_bundle'];bs=T/bf['relative_bundle']
            for b in [ss,bs]:validate_bundle(b);receipts[str(b/'receipt.json')]=file_hash(b/'receipt.json')
            qset={'original_RGMF':vz[f'old_rgmf_{seed}'],'strong_I0':vz[f'strong_baseline_{seed}'],'SIA_v2':np.load(ss/'predictions.npz')['quantiles'],'BAR':np.load(bs/'predictions.npz')['quantiles']}
            for model,q in qset.items():
                f=base.copy();f['seed']=seed;f['model']=model;f[COLS]=q;panel.append(f)
        folds[key]={'vz':vz,'bz':bz,'vm':vm,'bm':bm,'sia_fits':[f for f in sia if f['track']==track and f['year']==year],'bar_fits':[f for f in bar if f['track']==track and f['year']==year]}
    f=pd.concat(panel,ignore_index=True)
    for _,g in f.groupby(['track','model']):check_grid(g)
    return augment(f),folds,receipts

def features(folds):
    result=[];thresholds={};train_info={}
    for key,fold in folds.items():
        track,year=key.split(':');year=int(year);info='I3' if track=='Main-A' else 'I4';manifest,settings,panels,contexts,source=load_original(track)
        fs={i:track_folds(p,settings,track)[0][year] for i,p in panels.items()};common=set.intersection(*[set(zip(v['train'].loc[v['train'].context_eligible,'decision_time'],v['train'].loc[v['train'].context_eligible,'asset'])) for v in fs.values()]);tr=fs[info]['train'];tr=tr[[k in common for k in zip(tr.decision_time,tr.asset)]].sort_values(['decision_time','asset']);te=fs[info]['test'];cut=fs[info]['outer_cutoff'];schema=fold['vm']['schema']['schema'];names=[s['name'] for s in schema];contract=panels[info].attrs['neural_contract'];sources=contract['source_names']
        assert canonical_keys(zip(te.decision_time,te.asset))==canonical_keys(fold['bm']['date_asset_keys']);assert len(tr)==len(fold['vz']['tx']);assert (pd.to_datetime(tr.label_available_at,utc=True)<=cut).all();assert (pd.to_datetime(te.max_dependency_available_at,utc=True)<=pd.to_datetime(te.decision_time,utc=True)).all()
        tx=contexts[info][tr.sequence_index,-1,:];vx=contexts[info][te.sequence_index,-1,:];train_info[key]={'tr':tr,'te':te,'tx':tx,'vx':vx,'sources':sources,'names':names,'schema':schema,'cutoff':cut,'source':source}
        rows=pd.DataFrame(fold['bm']['date_asset_keys'],columns=['decision_time','asset']);rows['track']=track;rows['year']=year
        b={}
        for feature in ['realized_vol_20','return_4w','return_1w']:
            i=names.index(feature);assert schema[i]['kind']=='numeric' and schema[i]['source']=='market';bounds=np.quantile(tx[:,i][np.isfinite(tx[:,i])],[.25,.75]);rows[feature]=vx[:,i];rows[feature+'_regime']=tercile(vx[:,i],bounds);b[feature]={'bounds':bounds.tolist(),'training_distinct_dates':int(tr.decision_time.nunique()),'fit_cutoff':cut.isoformat(),'train_only':True,'source_column':i}
        magnitude=np.quantile(np.abs(tr.y),[.25,.75]);rows['OUTCOME_CONDITIONAL_magnitude']=tercile(np.abs(te.y),magnitude);b['realized_target_magnitude']={'bounds':magnitude.tolist(),'classification':'OUTCOME_CONDITIONAL_NOT_PREDICTIVE','train_only_bounds':True}
        for j,src in enumerate(sources):
            rows[src+'_available']=fold['bz']['test_valid'][:,j].astype(int);rows[src+'_OOD']=fold['bz']['test_ood'][:,j].astype(int)
            inds=[i for i,s in enumerate(schema) if s['kind']=='release_age' and s['source']==src];known=np.isfinite(vx[:,inds]);age=np.full(len(vx),np.nan)
            for row in range(len(vx)):
                if known[row].any():age[row]=np.max(vx[row,inds][known[row]])
            rows[src+'_release_age_max_days']=age
            vals=tx[:,inds][np.isfinite(tx[:,inds])];th=train_only_threshold(vals,np.repeat(tr.decision_time.to_numpy(),len(inds))[np.isfinite(tx[:,inds]).ravel()],np.repeat(tr.label_available_at.to_numpy(),len(inds))[np.isfinite(tx[:,inds]).ravel()],cut,cohort='MATURE_PRE_OUTER') if len(vals) else {'state':'NO_TRAIN_OBSERVED_SOURCE','thresholds':None}
            b[src+'_release_age_feature_warning']=th
            if th.get('thresholds'):rows[src+'_age_warning99']=np.where(np.isfinite(age),(age>th['thresholds']['0.99']).astype(int),-1)
            else:rows[src+'_age_warning99']=-1
        # Past-only baseline OOF is available; SIA/BAR corrected OOF is not.
        oof=fold['bz']['oof_native'];usable=np.isfinite(oof).all(1);assert (pd.to_datetime(tr.loc[usable,'label_available_at'],utc=True)<=cut).all();scale=tr.asset.map({a:float(fold['vz']['scale'][np.where(te.asset.to_numpy()==a)[0][0]]) for a in te.asset.unique()}).to_numpy();err=np.abs(tr.y.to_numpy()[usable]-oof[usable,2])/scale[usable];width=(oof[usable,4]-oof[usable,0])/scale[usable]
        for label,values in [('fixed_baseline_OOF_error',err),('fixed_baseline_OOF_width90',width)]:b[label]=train_only_threshold(values,tr.loc[usable,'decision_time'],tr.loc[usable,'label_available_at'],cut,cohort='MATURE_PRE_OUTER')
        wb=np.quantile(width,[.25,.75]);b['width_slice_bounds']={'bounds':wb.tolist(),'reference':'fixed baseline past-only OOF normalized widths','not_model_specific_error_calibration':True};rows['availability_count']=fold['bz']['test_valid'].sum(1);rows['any_source_OOD']=fold['bz']['test_ood'].any(1).astype(int)
        thresholds[key]=clean(b);result.append(rows)
    return pd.concat(result,ignore_index=True),thresholds,train_info

def calibration_and_slices(panel,thresholds):
    metrics={};quantiles=[];slices=[];high=[]
    prior=load(R/'reports/bar_exploratory_metrics_v1.json')['results']
    for (track,model),g in panel.groupby(['track','model']):
        key=f'{track}:{int(g.year.iloc[0])}';s=summary(g,ci=True);reference=prior[key]['BAR']['normalized_pinball'] if model=='BAR' else prior[key]['comparisons'][{'original_RGMF':'original_RGMF','SIA_v2':'SIA_v2','strong_I0':'strong_I0_baseline'}[model]]['old']['normalized_pinball'];assert np.isclose(s['normalized_pinball'],reference,rtol=1e-11);metrics[key+':'+model]=s
        for cohort,groups in [('ALL',[('ALL',g)]),('asset',g.groupby('asset')),('seed',g.groupby('seed'))]:
            for name,h in groups:
                d=summary(h)
                for j,q in enumerate(Q):quantiles.append({'track':track,'year':int(g.year.iloc[0]),'model':model,'cohort':cohort,'cohort_value':name,'quantile':q,'coverage':d['coverage'][j],'deviation_pp':d['coverage_deviation_pp'][j],'normalized_pinball':d['pinball_by_quantile'][j],'raw_pinball':d['raw_pinball_by_quantile'][j],'mean_signed_quantile_residual':float((h.target-h[COLS[j]]).mean()),'rows':len(h),'distinct_dates':h.decision_time.nunique(),'support':d['support']})
        local=g.copy();dates=pd.to_datetime(local.decision_time,utc=True);local['quarter']='Q'+dates.dt.quarter.astype(str);local['half_year']=np.where(dates.dt.month<=6,'H1','H2');local['interval_width_reference_bin']=tercile(local.normalized_width90,thresholds[key]['width_slice_bounds']['bounds']);fields=['asset','quarter','half_year','realized_vol_20_regime','return_4w_regime','return_1w_regime','OUTCOME_CONDITIONAL_magnitude','availability_count','any_source_OOD','interval_width_reference_bin']+[c for c in local if c.endswith('_available') or c.endswith('_age_warning99')]
        for field in fields:
            for value,h in local.groupby(field):
                d=summary(h);slices.append({'track':track,'year':int(g.year.iloc[0]),'model':model,'slice':field,'category':value,'classification':'OUTCOME_CONDITIONAL_NOT_PREDICTIVE' if field.startswith('OUTCOME_') else ('EX_ANTE_FEATURE_TRAIN_ONLY_BOUNDS' if field.endswith('_regime') else 'POSTHOC_DIAGNOSTIC_NOT_SELECTION'),'normalized_pinball':d['normalized_pinball'],'raw_pinball':d['raw_pinball'],'median_MAE':d['MAE'],'coverage90':d['coverage90'],'rows':len(h),'dates':d['distinct_dates'],'assets':d['assets'],'support':d['support']})
        weekly=local.groupby('decision_time').agg(normalized_pinball=('loss','mean'),raw_pinball=('raw_loss','mean'),median_MAE=('abs_error','mean'),max_abs_forecast=('max_abs_forecast','max')).sort_values('normalized_pinball',ascending=False)
        high += [dict(track=track,model=model,decision_time=dt,**r) for dt,r in weekly.head(10).to_dict('index').items()]
    return metrics,pd.DataFrame(quantiles),pd.DataFrame(slices),pd.DataFrame(high)

def paired_and_materiality(panel):
    attribution=[];ranks=[];pairs=[];material=[];exposures=[]
    for track,full in panel.groupby('track'):
        for a,b in itertools.combinations(MODELS,2):
            x=full[full.model.eq(a)].sort_values(['decision_time','asset','seed']);y=full[full.model.eq(b)].sort_values(['decision_time','asset','seed']);assert x[['decision_time','asset','seed']].values.tolist()==y[['decision_time','asset','seed']].values.tolist();diff=y[COLS].to_numpy()-x[COLS].to_numpy();d=x[['decision_time','asset','seed']].copy();d['loss_delta']=y.loss.to_numpy()-x.loss.to_numpy();d['median_abs_error_delta']=y.abs_error.to_numpy()-x.abs_error.to_numpy();d['sign_flip']=(np.sign(x.q50.to_numpy())!=np.sign(y.q50.to_numpy())).astype(int)
            pairs.append({'track':track,'a':a,'b':b,'median_sign_flip_fraction':float(d.sign_flip.mean()),'mean_quantile_abs_disagreement_native':np.abs(diff).mean(0).tolist(),'mean_quantile_abs_disagreement_normalized':(np.abs(diff)/x.scale.to_numpy()[:,None]).mean(0).tolist(),'loss_delta_b_minus_a':float(d.loss_delta.mean()),'mean_abs_median_error_change':float(d.median_abs_error_delta.mean()),'uncertainty':bootstrap_dates(d,['loss_delta','sign_flip']),'rows':len(d),'dates':52})
            for (dt,seed),h in x.groupby(['decision_time','seed']):
                z=y[(y.decision_time==dt)&(y.seed==seed)];ranks.append({'track':track,'a':a,'b':b,'decision_time':dt,'seed':seed,'asset_count':8,**rank_compare(h,z)})
            if b=='BAR' or a=='BAR':
                # Association with moving away from the target; never causal attribution.
                d['median_move_native']=diff[:,2];d['median_move_normalized']=diff[:,2]/x.scale.to_numpy();d['moved_away_from_target']=(d.median_abs_error_delta>0).astype(int);d['quantile_mean_disagreement']=np.abs(diff).mean(1);d['model_a']=a;d['model_b']=b;d['track']=track
                cov=x[['decision_time','asset','seed',*['realized_vol_20_regime','return_4w_regime','availability_count','any_source_OOD']]];d=d.merge(cov,on=['decision_time','asset','seed'],validate='one_to_one');attribution.append(d)
        for (model,dt,seed),g in full.groupby(['model','decision_time','seed']):
            g=g.sort_values('asset');a=g.asset.to_numpy();scores=g.q50.to_numpy()
            for mapping in ('top1','top2','linear_rank'):
                w=capped_weights(a,scores,mapping);commodity=np.array([x in COMMODITIES for x in a]);material.append({'track':track,'model':model,'decision_time':dt,'seed':seed,'mapping':mapping,'gross_signal_weight':float(w.sum()),'max_asset_weight':float(w.max()),'commodity_weight':float(w[commodity].sum()),'HHI_gross_weights':float((w*w).sum()),'top_asset':g.sort_values(['q50','asset'],ascending=[False,True]).asset.iloc[0],'scope':'P0_SIGNAL_PROXY_ONLY_NO_PNL'})
                exposures.extend([{'track':track,'model':model,'decision_time':dt,'seed':seed,'mapping':mapping,'asset':asset,'weight':float(weight)} for asset,weight in zip(a,w)])
    ex=pd.DataFrame(exposures);turn=[];changes=[]
    for (track,model,seed,mapping),g in ex.groupby(['track','model','seed','mapping']):
        pivot=g.pivot(index='decision_time',columns='asset',values='weight').sort_index();t=pivot.diff().abs().sum(1);t.iloc[0]=np.nan
        turn.append({'track':track,'model':model,'seed':seed,'mapping':mapping,'mean_L1_signal_weight_change':float(t.iloc[1:].mean()),'dates':len(pivot),'transitions':len(pivot)-1,'PnL':None,'P1_status':'BLOCKED_P1'})
    for track,g in ex.groupby('track'):
        for model in ['original_RGMF','SIA_v2','BAR']:
            a=g[g.model.eq('strong_I0')];b=g[g.model.eq(model)];z=a.merge(b,on=['track','decision_time','seed','mapping','asset'],suffixes=('_baseline','_candidate'),validate='one_to_one');z['L1']=abs(z.weight_baseline-z.weight_candidate)
            for mapping,h in z.groupby('mapping'):changes.append({'track':track,'candidate':model,'mapping':mapping,'mean_L1_vs_baseline':float(h.groupby(['decision_time','seed']).L1.sum().mean()),'P0_only':True})
    return clean(pairs),pd.DataFrame(ranks),pd.concat(attribution,ignore_index=True),pd.DataFrame(material),ex,pd.DataFrame(turn),pd.DataFrame(changes)

def param_and_ablation(folds):
    params=[];cosines=[];ablation=[];gates=[];checks=[]
    for key,fold in folds.items():
        track,year=key.split(':');vectors={}
        for family,rows in [('SIA_v2',fold['sia_fits']),('BAR',fold['bar_fits'])]:
            vs=[]
            for fit in rows:
                b=T/fit['relative_bundle'];state=torch.load(b/'model.pt',map_location='cpu',weights_only=False);weights=state['state'] if family=='BAR' else state['model'];flat=np.concatenate([v.detach().numpy().astype(float).ravel() for k,v in sorted(weights.items())]);vs.append((fit['seed'],flat,tuple((k,tuple(v.shape)) for k,v in sorted(weights.items()))));head={k:float(torch.linalg.vector_norm(v.float())) for k,v in weights.items() if 'head' in k or 'gate' in k};params.append({'track':track,'year':int(year),'family':family,'seed':fit['seed'],'parameter_tensor_elements':len(flat),'L2_norm':float(np.linalg.norm(flat)),'max_abs_parameter':float(np.abs(flat).max()),'head_gate_tensor_norms':head,'selected_epoch':state.get('selected_epoch'),'trained_epochs':len(state.get('losses',[])) if family=='SIA_v2' else state['selected_epoch']})
                if family!='BAR':continue
                model=BAR(**state['spec']);model.load_state_dict(weights);model.eval();z=fold['bz'];md=fold['bm'];groups=md['groups'];sources=sorted(set(s['source'] for s in fold['vm']['schema']['schema'])-{'market'});assert len(sources)==len(groups)
                def predict(valid,ood):
                    with torch.no_grad():
                        q=model(torch.as_tensor(z['test_base']/z['scale'][:,None],dtype=torch.float64),source_tensors(z['test_x'],groups,device='cpu'),torch.as_tensor(valid),torch.as_tensor(ood)).numpy()*z['scale'][:,None]
                    fallback=~(valid&~ood).any(1);q[fallback]=z['test_base'][fallback];return q
                allq=predict(z['test_valid'],z['test_ood']);saved=np.load(b/'predictions.npz')['quantiles'];maxdelta=float(np.abs(allq-saved).max());replay=replay_status(allq,saved);checks.append({'fit_id':fit['fit_id'],'CPU_vs_saved_CUDA_max_abs_difference':maxdelta,'tolerance_atol':2e-7,'tolerance_rtol':2e-6,'training':False,'replay_status':replay})
                base=allq;target=z['target'];scale=z['scale'];loss=lambda q:np.maximum(Q*(target[:,None]-q),(Q-1)*(target[:,None]-q)).mean(1)/scale
                all_missing=predict(np.zeros_like(z['test_valid']),z['test_ood']);assert np.array_equal(all_missing,z['test_base'])
                tensors=source_tensors(z['test_x'],groups,device='cpu');rel=[];saturation=[]
                with torch.no_grad():
                    for j,(enc,gate,x) in enumerate(zip(model.encoders,model.gates,tensors)):
                        active=z['test_valid'][:,j]&~z['test_ood'][:,j];cx=torch.where(torch.as_tensor(active)[:,None,None],x,torch.zeros_like(x));latent=enc(cx)[0][:,-1];gw=(torch.sigmoid(gate(latent)).squeeze(1)/len(groups)).numpy();raw=model.residual.heads[j](latent).numpy();sat=np.abs(np.tanh(raw[:,1:]));gates.append({'track':track,'seed':fit['seed'],'source':sources[j],'availability_fraction':float(active.mean()),'mean_effective_gate':float((gw*active).mean()),'gate_p99':float(np.quantile(gw[active],.99)) if active.any() else None,'head_saturation_gt095':float((sat[active]>.95).mean()) if active.any() else None})
                for j,src in enumerate(sources):
                    valid=z['test_valid'].copy();valid[:,j]=False;q=predict(valid,z['test_ood']);d=np.abs(q-base)/scale[:,None];ablation.append({'track':track,'year':int(year),'seed':fit['seed'],'masked_source':src,'normalized_pinball_full':float(loss(base).mean()),'normalized_pinball_masked':float(loss(q).mean()),'delta_masked_minus_full':float((loss(q)-loss(base)).mean()),'mean_abs_quantile_change_normalized':float(d.mean()),'max_abs_quantile_change_normalized':float(d.max()),'available_rows':int(z['test_valid'][:,j].sum()),'rows':416,'scope':'FROZEN_MODEL_PERTURBATION_NOT_CAUSAL'})
            for (sa,a,shapea),(sb,b,shapeb) in itertools.combinations(vs,2):
                assert shapea==shapeb;cosines.append({'track':track,'family':family,'seed_a':sa,'seed_b':sb,'cosine_same_parameter_coordinates':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),'interpretation':'Same architecture only; neuron permutation means low cosine is not functional instability'})
    return clean(params),pd.DataFrame(cosines),pd.DataFrame(ablation),pd.DataFrame(gates),checks
def methods_and_risks():
    methods=[
      ['PHT/semantic age inspiration','SIA implemented and6 real CUDA fits','Fixed log1p release/reference ages, explicit binary masks/constant-column OOD; full published-PHT reproduction not established','src/signalforge/v33_sia.py; reports/v33_minimal_integrity_v2.json'],
      ['Baseline/physics-inspired residual','BAR implemented and6 real CUDA fits; negative vs strong controls','Statistical anchoring, not physical equations/PDE constraints; old v2 BAR gate unchanged','studies_v3/bar_model.py; configs/bar_post_result_exploratory_v1r1.json; reports/bar_fits_v1.json'],
      ['ABC asymmetric source correction','SYNTHETIC_PROTOTYPE_ONLY','No real fits or calibration, no tested investment alpha','src/signalforge/v33_corrections.py; tests/v33/test_corrections.py'],
      ['Availability masks / SIA OOD','IMPLEMENTED_AND_EXECUTED','Typed raw-value validity; OOD detects train-constant/unseen numeric columns only, not comprehensive distributional OOD','src/signalforge/track_neural.py; src/signalforge/v33_sia.py'],
      ['Mature prequential baseline OOF','IMPLEMENTED_AND_EXECUTED','441/66 residual OOF dates; no SIA/BAR model-specific mature OOF predictions','reports/bar_oof_evidence_v1.json; prepared bundles'],
      ['Zero heads / bounded residual / quantile monotonicity / fallback','IMPLEMENTED_TESTED_EXECUTED','CPU perturbation validates saved BAR; fallback mechanically tested, not a production qualification','studies_v3/bar_model.py; reports/bar_independent_validation_v1.json'],
      ['Reliability/gating / source bias','LEARNED_GATES_EXECUTED; DIAGNOSTICS_NOW_EXECUTED','Gates are not calibrated source truth probabilities; source perturbation is not causal ablation/retraining','reports/model_risk_v3/source_ablation.csv; gate_diagnostics.csv'],
      ['Quantile-tail calibration for SIA/BAR','NOT_FITTED_BLOCKED_OOF_UNAVAILABLE','Outer evaluation outputs cannot train calibrators; baseline OOF is not corrected model OOF','reports/bar_exploratory_metrics_v1.json; monitoring.json'],
      ['Parameter/seed stability / challenger checks','POSTHOC_EXECUTED','Same architecture seed cosine only; neuron permutation makes weight cosine non-identifiable','reports/model_risk_v3/parameter_stability.json; challenger_ranks.csv'],
      ['36-fit / ABC / full HPO / SSL / v3.2 expansion','NOT_EXECUTED_IN_THIS_STUDY','Original completed experiments remain distinct; user stopped36-fit programme; BAR negative expansion rule maintained','configs/v33_retrospective_diagnostic.json; reports/bar_final_handoff_v1.json'],
      ['P0 portfolio materiality','SIGNAL_PROXY_EXECUTED','Ranks/capped weight changes only; actual orders/fills/costs/PnL absent','reports/model_risk_v3/signal_materiality.csv'],
      ['Tier-A / P1 / reserved final / prospective','UNQUALIFIED_NOT_EXECUTED','No original-publication certification, economic execution qualification, final unsealing or future outcomes','reports/bar_final_handoff_v1.json']]
    risks=[
      ['R01','HIGH','Tail miscalibration','Coverage below/above intended tails; pooled rows exaggerate support','Mature-model OOF unavailable; quarantine calibrated claims','Research owner','Review before any promotion; fit future calibrator only on qualified OOF'],
      ['R02','HIGH','Strong baseline degradation','BAR fails strong baselines on both tracks','Registered expansion stopped, keep baseline challenger','Research owner','STOP expansion; separate rationale required'],
      ['R03','HIGH','Post-hoc selection / many slices','Audit sees known outcomes, many dependent comparisons','No new selection/threshold optimization; descriptive CIs only','Validation owner (not independent reviewer)','Document and replicate on separately authorized independent data'],
      ['R04','HIGH','P0/P1 mismatch','Signal ranks do not establish executed PnL','BLOCKED_P1 no economic metrics','Data/economics owner','STOP economic claims until actual open/action/pay-date/cash evidence'],
      ['R05','HIGH','Tier-B source clock','Reconstructed availability is not original-publication proof','Strict final sealed','Source owner','STOP strict-PIT/final promotion'],
      ['R06','MEDIUM','Release-age/scaling explosion','Old nearlyconstant learned age std risk; SIA fixed semantic ages','Finite/schema/age guard; fixed log1p','Runtime owner','Quarantine malformed/future/negative-age inputs'],
      ['R07','MEDIUM','Rare sources / masks','EIA applies onlyUSO; NPORT coverage onlyregistered ETFs','Train-only cohort counts; no invented UNG EIA','Source owner','Review sparse coverage; UNDERPOWERED cohorts do not qualify'],
      ['R08','MEDIUM','Seed/parameter stability','Random latent permutations and training trajectories','Prediction/loss spread primary, cosine descriptive','Model owner','Review disagreement without selecting favorable seed'],
      ['R09','HIGH','OOF survivorship/support','Earliest26 baseline dates omitted by min support, residual subset maturity purge','Full evaluation retention; OOF counts and lineage recorded','Validation owner','Review support; no relaxed maturity'],
      ['R10','MEDIUM','Regime drift / limited OOD','Constant-column OOD not tail/variance shift detector','Train-only warning candidates; no controlled FPR claim','Monitoring owner','Review drift with fresh mature observations'],
      ['R11','HIGH','Absent independent governance','Developer executed audit, no external challenge signoff','No regulatory compliance assertion','Governance owner UNASSIGNED','Require independent reviewer before operational deployment']]
    return pd.DataFrame(methods,columns=['method','actual_status','limit','evidence']),pd.DataFrame(risks,columns=['risk_id','severity','failure_mode','evidence_or_risk','mitigation','proposed_owner','action'])

def monitoring(thresholds,train_info,panel):
    rows=[{'indicator':x,'threshold':0,'source':'NORMATIVE_SCIENTIFIC_GUARD','status':'NORMATIVE_NOT_STATISTICALLY_CALIBRATED','cadence':'each input/prediction batch','owner':'runtime/data owner (proposed)','action':'STOP/quarantine; preserve receipt, investigate before fallback','page_or_ticket':'PAGE_SCIENTIFIC_INTEGRITY'} for x in ['nonfinite forecasts or target','quantile crossings','future feature availability','immature training labels','schema/source/hash mismatch','reserved cohort access']]
    rows += [{'indicator':'max_abs_forecast>100','threshold':100,'source':'existing numerical catastrophe convention, not calibrated investment risk','status':'NORMATIVE_DIAGNOSTIC','cadence':'each forecast','owner':'model owner (proposed)','action':'quarantine and investigate; baseline fallback only if qualified','page_or_ticket':'PAGE_RUNTIME'}, {'indicator':'OOD/availability changes','threshold':None,'source':'training distributions recorded; operating minimum requires source-specific contract','status':'PROPOSED_NEEDS_NEW_DATA','cadence':'each batch/weekly','owner':'source owner (proposed)','action':'ticket persistent change; inspect lineage; no universal availability threshold','page_or_ticket':'TICKET_SOURCE'}, {'indicator':'PSI/KS distribution drift','threshold':None,'source':'not computed as formal FPR-controlled detector','status':'PROPOSED_NEEDS_NEW_DATA','cadence':'monthly with mature sample support','owner':'validation owner (proposed)','action':'review; no auto-promotion','page_or_ticket':'REVIEW_DRIFT'}, {'indicator':'SIA/BAR matured pinball / tail coverage / disagreement model-specific error limits','threshold':None,'source':'model-specific mature OOF missing','status':'BLOCKED_OOF_UNAVAILABLE','cadence':'when genuinely matured forward labels available','owner':'validation owner (proposed)','action':'retain baseline challenger; no outer-year tuning','page_or_ticket':'REVIEW_NOT_AUTOMATIC_CALIBRATION'}]
    breaches=[]
    for key,t in thresholds.items():
        track=key.split(':')[0];g=panel[panel.track.eq(track)]
        for label,th in t.items():
            if not isinstance(th,dict) or not th.get('thresholds'):continue
            rows.append({'track':track,'indicator':label,'threshold':th['thresholds'],'training_dates':th['distinct_dates'],'source':'pre-outer mature training; fixed baseline OOF only for baseline error/width','status':th['state'],'operational_freeze':False,'cadence':'weekly source and later mature baseline outcomes','owner':'monitoring owner UNASSIGNED','action':'illustrative ticket/review candidate; no trading/production threshold authorization','page_or_ticket':'ILLUSTRATIVE_REVIEW'})
        th=t['fixed_baseline_OOF_error']
        for model,h in g.groupby('model'):
            limit=th['thresholds']['0.99'];breaches.append({'track':track,'model':model,'baseline_reference_error_limit':limit,'rows_exceeding':int((h.normalized_abs_error>limit).sum()),'dates_any_exceeding':int(h.loc[h.normalized_abs_error>limit,'decision_time'].nunique()),'status':'ILLUSTRATIVE_RETROSPECTIVE_NOT_MODEL_CALIBRATED'})
    return {'catalog':rows,'train_only_candidates':thresholds,'retrospective_reference_breaches':breaches,'model_specific_error_thresholds':{m:model_error_threshold(m,False) for m in ['SIA_v2','BAR','original_RGMF']},'calibrator_fits':0,'operationally_frozen':False,'owners_are_proposed_not_actual_appointments':True,'independent_external_governance_review':False}

def independent(panel,folds,receipts,before):
    # Separate scalar calculation from vectorized core metrics.
    scores={}
    for (track,model),g in panel.groupby(['track','model']):
        check_grid(g);daily=[]
        for dt,day in g.groupby('decision_time'):
            asset_scores=[]
            for asset,rows in day.groupby('asset'):
                seeds=[]
                for row in rows.itertuples():
                    y=float(row.target);s=float(row.scale);losses=[]
                    for c,tau in zip(COLS,[.05,.1,.5,.9,.95]):
                        e=y-float(getattr(row,c));losses.append(tau*e/s if e>=0 else (tau-1)*e/s)
                    seeds.append(sum(losses)/5)
                asset_scores.append(sum(seeds)/len(seeds))
            daily.append(sum(asset_scores)/len(asset_scores))
        score=sum(daily)/len(daily);assert abs(score-summary(g)['normalized_pinball'])<1e-12;scores[track+':'+model]=score
    for p,h in receipts.items():assert file_hash(Path(p))==h
    after=preserve();assert before==after
    return {'state':'PASSED_INDEPENDENT_POSTHOC_NUMERICS','scores':scores,'rows_per_model':2496,'rows_total_four_models':9984,'dates_per_track':52,'source_hash':code_hash(R),'diagnostic_code_hash':diag_hash(),'old_v2_and_BAR_and_original_preserved':True,'new_GPU_fits':0,'calibrator_fits':0,'reserved_access':False,'qualified_for_final':False,'historical_suite_artifact':'105 passed1 failed initial-v1/current-v2 source identity assertion; unchanged','preservation':after}

def plots(metrics,slices,ablation):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,track in zip(axs,['Main-A:2020','Nested-B:2022']):
        for m in MODELS:ax.plot(Q,metrics[track+':'+m]['coverage'],marker='o',label=m)
        ax.plot(Q,Q,'k--',lw=1);ax.set(title=track,xlabel='Nominal quantile',ylabel='Empirical coverage',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
    fig.suptitle('Post-hoc development only:52 dates per track');fig.tight_layout();fig.savefig(O/'quantile_coverage.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,track in zip(axs,['Main-A','Nested-B']):
        h=slices[(slices.track==track)&(slices['slice']=='asset')].pivot(index='category',columns='model',values='normalized_pinball');h[MODELS].plot.bar(ax=ax);ax.set(title=track,ylabel='Normalized pinball');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(O/'asset_loss.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,track in zip(axs,['Main-A','Nested-B']):
        h=ablation[ablation.track==track].groupby('masked_source').delta_masked_minus_full.mean();h.plot.bar(ax=ax);ax.axhline(0,color='black',lw=.7);ax.set(title=track,ylabel='Masked minus full loss (CPU)')
    fig.suptitle('Frozen BAR source perturbation, not causal effect');fig.tight_layout();fig.savefig(O/'source_sensitivity.png',dpi=150);plt.close(fig)

def report(metrics,slices,ranks,params,ablation,gates,material,turnover,changes,risks,monitoring_receipt,methods,validation,high):
    lines=['# v3.3 comprehensive post-hoc model-risk audit','','Executed developer outcomes/monitoring/materiality analysis using immutable development predictions. No training, calibration fit, provider acquisition, execution prices, reserved access, model promotion, or final claims. This is not regulatory compliance or an external independent governance review.','', 'Model-risk framing: [Federal Reserve SR26-2, April17 2026](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) supersedes SR11-7. Here conceptual soundness, outcomes analysis, ongoing monitoring, effective challenge and materiality are analytical categories only.','', '## Actual methods versus missing methods','', '| Method | Actual status | Boundary |','|---|---|---|']
    for row in methods.to_dict('records'):lines.append(f"| {row['method']} | {row['actual_status']} | {row['limit']} |")
    lines += ['','## Matched forecast reproduction','','52 market dates ×8 assets ×3 seeds per track/family;1,248 rows per track and2,496 per model,9,984 across4 families. Same date/asset/seed/target/scale/fold/source identities. Seeds/assets are correlated repeats. No favorable exclusions. Normalized loss is date-equal, asset-equal, seed-averaged. Raw loss is separate.','', '| Track/model | NPL | Raw pinball | MAE | q05/q10/q50/q90/q95 coverage | 90% coverage | 90% raw width |','|---|---:|---:|---:|---|---:|---:|']
    for k,s in metrics.items():lines.append(f"| {k} | {s['normalized_pinball']:.8f} | {s['raw_pinball']:.8f} | {s['MAE']:.6f} | {','.join(f'{v:.4f}' for v in s['coverage'])} | {s['coverage90']:.4f} | {s['width90']:.6f} |")
    lines+=['','## Calibration, concentration, regime and error findings','','Date-block moving bootstrap4/8/13 weeks with2,000 draws covers primary loss, quantile coverage, interval coverage/width and normalized MAE. These intervals are descriptive; multiple slices are not a confirmatory hypothesis family and no significance/null claims are made. Quantile/asset/seed coverage deviations and pinball are in quantile_calibration.csv; interval scores, bias, p95/p99 loss and concentration in metrics.json.']
    for key in ['Main-A:2020','Nested-B:2022']:
        b=metrics[key+':BAR'];s=metrics[key+':SIA_v2'];i=metrics[key+':strong_I0'];lines += ['',f"{key}: BAR vs strongest loss change {(b['normalized_pinball']/i['normalized_pinball']-1)*100:.4f}%; vs SIA {(b['normalized_pinball']/s['normalized_pinball']-1)*100:.4f}%. BAR q0.95 deviation {(b['coverage'][4]-.95)*100:.3f}pp and q0.10 deviation {(b['coverage'][1]-.10)*100:.3f}pp. Numerical crossings/catastrophes0 does not establish tail calibration. Largest absolute forecast {b['max_abs_forecast']:.6f}.",f"BAR loss contribution USO {b['asset_loss_fractions'].get('USO',0)*100:.2f}%, UNG {b['asset_loss_fractions'].get('UNG',0)*100:.2f}%; top5 market dates contribute {b['top5_date_loss_fraction']*100:.2f}% of date-equal loss. These assets/dates remain included."]
        t=key.split(':')[0];rs=slices[(slices.track==t)&(slices.model=='BAR')&(slices['slice']=='realized_vol_20_regime')];lines.append('Ex-ante historical20-session volatility, train-only quartile boundaries: '+ '; '.join(f"{r.category} loss {r.normalized_pinball:.6f}, {r.dates}dates, {r.support}" for r in rs.itertuples())+'.')
    lines += ['', 'Regimes use exact registered return_1w, return_4w and realized_vol_20 raw features already available at decision time; bounds from pre-outer mature training only. Realized target-magnitude cohorts are OUTCOME_CONDITIONAL_NOT_PREDICTIVE, not market regime predictions. Quarter/half-year, source masks/ages, width reference bins and per-asset slices retained. Cohorts<13 dates are UNDERPOWERED. EIA source is USO activity, not UNG natural gas; rare masks and mapping are not arbitrary data omissions.','', '## Parameters, seeds and frozen-model source sensitivity','','Six SIA and six BAR model states read on CPU; tensor/head/gate norms, parameter element counts and same-architecture seed cosines recorded. Neuron permutation means raw cosine is not identifiable functional stability. CPU-vs-saved-CUDA strict replay tolerance failed for some fits; declared atol2e-7/rtol2e-6 was not relaxed. CPU_backend_discrepancy.json preserves errors; CPU perturbations are descriptive backend-specific sensitivities, not qualified exact CUDA replay. All-source-missing fallback equals anchor bitwise. No fit or CUDA call.']
    for track in ['Main-A','Nested-B']:
        h=ablation[ablation.track==track].groupby('masked_source').delta_masked_minus_full.mean();lines += ['',track+' masked-source minus full NPL: '+', '.join(f'{src} {v:+.8f}' for src,v in h.items())+'. Positive means masking hurt, negative means masking helped. Frozen perturbation is association/sensitivity, not causal source value or retrained ablation.']
        rr=ranks[(ranks.track==track)&(ranks.a=='strong_I0')&(ranks.b=='BAR')];lines.append(f"BAR vs strongest: top1 switch {rr.top1_switch.mean()*100:.2f}%, top2 Jaccard {rr.top2_jaccard.mean():.4f}, mean Spearman {rr.spearman.mean():.4f}. Each rank uses only8 assets; ties and seed repetitions recorded.")
    lines += ['', '## Challenger attribution and P0 materiality','','All six model pairings have per-quantile native/normalized disagreement, sign flips, weekly Spearman/Kendall/ties, top1 switches/top2 Jaccard and date-block uncertainty. BAR comparisons contain row-level changes in median error and loss with source/volatility tags; largest disagreements and weekly errors are retained. A median sign is a P0 close-benchmark target sign, not an execution signal.','', 'Three fixed transparent signal-weight mappings: top1, top2 and linear rank. Long-only hypothetical rank weights capped per asset0.25, commodity total0.35 (GLD/SLV/UNG/USO), gross<=1, remainder unallocated. Weight changes are signal turnover proxies, not actual trades, costs or profitability. No PnL/Sharpe/alpha or target-weight return calculation.']
    for r in changes[changes.candidate=='BAR'].itertuples():lines.append(f"{r.track} {r.mapping}: mean L1 rank-weight difference from strongest {r.mean_L1_vs_baseline:.6f}.")
    lines += ['', '## Train-only threshold candidates and monitoring','','Release-age quantiles and baseline OOF error/width p95/p99 are computed on pre-outer mature training only, with cutoff/support/feature and OOF lineage. These are candidates, NOT operational/scientific freezes or controlled-FPR limits. All observed retrospective breaches are illustrative. Baseline OOF belongs to fixed historical / fixed linear-quantile alpha0.1, not SIA/BAR and not the old Nested alpha0.088586679 control. SIA/BAR/original corrected-model calibration/error thresholds: BLOCKED_OOF_UNAVAILABLE. No calibrator fitted.','Normative integrity hard stops: nonfinite, crossing, future feature, immature label, wrong schema/hash, reserved access. Source-age/availability/drift/disagreement/width warnings require named owner, support/cadence and escalation; proposed owners are not actual governance appointments. Routine review must examine mature performance and source changes; fallback itself requires validated baseline inputs. PSI/KS thresholds not calibrated here and no FPR claim. Full catalog/risk register is in monitoring.json and risk_register.csv.','', '## Verification, errors and genuine blockers','','23 version-specific diagnostic tests executed; actual test log in tests.json. Scalar independent date→asset→seed→quantile recomputation reproduces vectorized loss; all9,984 rows retained. Frozen v2 code_hash remains471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870. Original357 protected files/three original ledgers, v2 pilot/full ledgers, BAR observation databases and immutable completion hashes verified. Diagnostic code separately hashed; no src/scripts or old report/config changes.','The earlier expanded historical suite remains105 pass1 fail: initial-v1 plan insists on its earlier source hash instead of completed-v2 source. Archived test/plan/receipt unchanged; no all-suite-green claim.','Remaining: mature corrected-model OOF for calibration, independent future replication/monitoring cohort, real independent governance reviewer, authenticated Tier-A clocks, qualified P1 raw opens/actions/ex/pay/splits/cash and final freeze/authorization. No model promotion follows this audit; BAR negative expansion decision preserved.','', '## Data dictionary and reproduction','','panel.csv (ext4 immutable bundle): track/year/model/date/asset/seed, five native quantiles, native target, train-only scale, IDs, pinball/error/coverage/interval diagnostics, decision-known regimes/source masks/ages. CSVs in this report folder contain aggregate slices or perturbation results. quantile_calibration: deviation pp is empirical minus nominal. source_ablation: masked minus full loss, no refitting. signal_weights: capped P0 proxy only. thresholds: historical pre-outer candidates, not deployment authority. bootstrap intervals: market-date blocks only.','Run from isolated root: `pwsh -NoProfile -File diagnostics_v3/Invoke-ModelRisk.ps1 -Action tests` then `-Action run`. All completed diagnostic outputs bind hashes. Reproduction refuses inconsistent frozen inputs and never retrains models.']
    (O/'V33_COMPREHENSIVE_POSTHOC_MODEL_RISK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    torch.set_num_threads(4);O.mkdir(parents=True,exist_ok=True);event('PREFLIGHT_IMMUTABLE_READ_ONLY');before=preserve();tests();event('DIAGNOSTIC_TESTS_PASSED');panel,folds,receipts=load_forecasts();event('EXACT_FOUR_FAMILY_PANEL_VERIFIED',rows=len(panel),rows_per_model=2496)
    feat,thresholds,train_info=features(folds);panel=panel.merge(feat,on=['track','year','decision_time','asset'],validate='many_to_one');assert len(panel)==9984;save('threshold_candidates',thresholds);feat.to_csv(O/'ex_ante_features.csv',index=False)
    event('DECISION_KNOWN_REGIMES_AND_TRAIN_ONLY_CANDIDATES_READY');metrics,quantiles,slices,high=calibration_and_slices(panel,thresholds);save('metrics',metrics);quantiles.to_csv(O/'quantile_calibration.csv',index=False);slices.to_csv(O/'slices.csv',index=False);high.to_csv(O/'high_error_dates.csv',index=False)
    # Exact matched slice deltas against strong and SIA, retaining sparse support.
    a=slices[slices.model.eq('strong_I0')].drop(columns='model');b=slices[slices.model.eq('SIA_v2')].drop(columns='model');join=['track','year','slice','category'];sd=slices.merge(a[join+['normalized_pinball']],on=join,suffixes=('','_strong'),validate='many_to_one').merge(b[join+['normalized_pinball']],on=join,suffixes=('','_SIA'),validate='many_to_one');sd['delta_vs_strong']=sd.normalized_pinball-sd.normalized_pinball_strong;sd['delta_vs_SIA']=sd.normalized_pinball-sd.normalized_pinball_SIA;sd.to_csv(O/'slice_deltas.csv',index=False)
    event('CALIBRATION_AND_ERROR_SLICES_COMPLETE');pairs,ranks,attribution,material,exposure,turnover,changes=paired_and_materiality(panel);save('challenger_pairs',pairs)
    for name,df in [('challenger_ranks',ranks),('BAR_disagreement_attribution',attribution),('signal_materiality',material),('signal_weights',exposure),('signal_turnover',turnover),('signal_materiality_vs_baseline',changes)]:df.to_csv(O/(name+'.csv'),index=False)
    attribution.nlargest(60,'quantile_mean_disagreement').to_csv(O/'largest_disagreements.csv',index=False);event('CHALLENGER_AND_P0_MATERIALITY_COMPLETE',economic_claims=False)
    params,cosines,ablation,gates,checks=param_and_ablation(folds);save('parameter_stability',params);save('CPU_inference_verification',checks);cosines.to_csv(O/'seed_parameter_cosines.csv',index=False);ablation.to_csv(O/'source_ablation.csv',index=False);gates.to_csv(O/'gate_diagnostics.csv',index=False);event('CPU_FROZEN_SOURCE_ABLATIONS_COMPLETE',masked_inferences=len(ablation),new_fits=0)
    methods,risks=methods_and_risks();methods.to_csv(O/'method_implementation_audit.csv',index=False);risks.to_csv(O/'risk_register.csv',index=False);mon=monitoring(thresholds,train_info,panel);save('monitoring',mon);plots(metrics,slices,ablation)
    # Immutable ext4 panel and exact schema/source identities, no original writes.
    csv=panel.to_csv(index=False).encode();schema={'columns':{c:str(panel[c].dtype) for c in panel},'rows':len(panel),'dates_per_track':52,'source_receipts':receipts,'diagnostics_hash':diag_hash(),'frozen_v2_source_hash':code_hash(R),'input_config_sha256':{p:file_hash(R/p) for p in ['configs/v33_minimal_execution_v2r1.json','configs/bar_post_result_exploratory_v1r1.json','reports/bar_frozen_plan_v1.json']},'reserved_access':False};identity=digest({'schema':schema,'panel_sha256':__import__('hashlib').sha256(csv).hexdigest()});dest=T/'artifacts/model_risk_v3'/identity;commit_bundle(dest,{'panel.csv':csv,'schema.json':schema},panel_bundle_metadata(diag_hash()));validate_bundle(dest);save('panel_receipt',{'relative_bundle':str(dest.relative_to(T)),'schema':schema,'receipt_sha256':file_hash(dest/'receipt.json')})
    validation=independent(panel,folds,receipts,before);save('independent_validation',validation);report(metrics,slices,ranks,params,ablation,gates,material,turnover,changes,risks,mon,methods,validation,high)
    # Bind all finished outputs before final status; report execution, not scientific qualification.
    files={str(p.relative_to(R)):file_hash(p) for p in sorted(O.iterdir()) if p.is_file() and p.name not in ['live_progress.json','completion.json']};completion={'state':'COMPLETED_POSTHOC_DEVELOPER_MODEL_RISK_AUDIT','diagnostic_code_hash':diag_hash(),'frozen_source_hash':code_hash(R),'files':files,'panel_relative_bundle':str(dest.relative_to(T)),'GPU_training_fits':0,'calibrator_fits':0,'model_promotions':0,'qualified_for_final':False,'reserved_access':False,'blocks':['corrected model OOF unavailable','P1 unqualified','TierA/final sealed','independent reviewer absent','CPU exact CUDA replay tolerance failed; CPU sensitivities descriptive only'],'historical_test_failure_preserved':True};save('completion',completion);event('POSTHOC_MODEL_RISK_COMPLETED',models=4,forecast_rows=9984,new_GPU_fits=0);print(json.dumps({'scores':validation['scores'],'report':str(O/'V33_COMPREHENSIVE_POSTHOC_MODEL_RISK.md')}),flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1]=='tests':tests()
        else:main()
    except BaseException:
        import traceback
        save('failure_'+now().replace(':','_'),{'state':'FAILED_DIAGNOSTICS','traceback':traceback.format_exc(),'new_training_fits':0,'reserved_access':False});raise
