"""Measured resumable development execution for the reconstructed Auxiliary branch.

This branch cannot stand in for Main/Nested information comparisons or qualify
the reserved study. It retains failures, forecasts, transforms and fitted models.
"""
import json
from pathlib import Path
import tempfile
import time
import shutil
from contextlib import nullcontext
import numpy as np
import pandas as pd
from .auxiliary import FAMILIES, sequences, fit_predict
from .data import auxiliary_rows
from .features import TrainTransform, shared_target_scale, purged_training
from .models import QUANTILES,NeuralCUDA
from .metrics import quantile_loss
from .experiments import TrialLedger, ComputeBudget
from .runtime import atomic_json, commit_bundle, validate_bundle, digest, code_hash, gpu_lease, now
from .runtime import admission

CPU_FAMILIES={'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}


def trial_grid(family):
    if family in {'historical','ewma'}:return [{'regularization':1.,'neural':None}]
    if family in CPU_FAMILIES|{'lightgbm','xgboost','lgb_no_age','lgb_no_coverage'}:
        return [{'regularization':float(v),'neural':None} for v in np.geomspace(.0001,10000,20)]
    return [{'regularization':1.,'neural':[width,lr]} for width in [8,16,32,48,64] for lr in [.0003,.001,.003,.01]]


def preprocess(x,training,testing,cutoff):
    tx=x[training.sequence_index];vx=x[testing.sequence_index]
    dates=np.repeat(training.decision_time.to_numpy(),tx.shape[1])
    transform=TrainTransform().fit(tx.reshape(-1,tx.shape[-1]),dates,cutoff)
    ft=transform.transform(tx.reshape(-1,tx.shape[-1])).reshape(len(tx),tx.shape[1],-1)
    fv=transform.transform(vx.reshape(-1,vx.shape[-1])).reshape(len(vx),vx.shape[1],-1)
    scale,normalizer=shared_target_scale(training.y.to_numpy(),training.label_available_at,cutoff)
    return ft,fv,scale,normalizer,transform


def fit_artifact(repo,runtime,budget,family,trial,seed,training,testing,arrays,identity,code,data_id,reservation=300,raw_sequences=None):
    directory=runtime/'artifacts/auxiliary'/identity
    if (directory/'receipt.json').exists():
        validate_bundle(directory)
        return json.loads((directory/'metrics.json').read_text())
    tx,vx,scale,normalizer,transform=arrays
    feature_variant=None
    if family in {'lgb_no_age','lgb_no_coverage'}:
        tx,vx=tx.copy(),vx.copy();f=tx.shape[-1]//2
        columns=[10,11,f+10,f+11] if family=='lgb_no_age' else list(range(f,2*f))
        tx[...,columns]=0;vx[...,columns]=0
        feature_variant={'kind':family,'removed_transformed_columns':columns,'retrained':True,'intervention_type':'retrained_ablation'}
    resource={}
    if family not in CPU_FAMILIES:
        import torch
        import psutil
        host_path=repo/'reports/host_disk_evidence.json'
        host=json.loads(host_path.read_text()) if host_path.exists() else {}
        fresh=pd.Timestamp.now(tz='UTC')-pd.Timestamp(host.get('observed_at','2000-01-01T00:00Z'))<pd.Timedelta(hours=2)
        free,total=torch.cuda.mem_get_info()
        # Conservative small-panel allocations; admission is based on actual free capacity.
        ram_estimate=max(512*1024**2,tx.nbytes*20);vram_estimate=max(1024**3,tx.nbytes*30)
        if not admission(ram_estimate,vram_estimate,64*1024**2,psutil.virtual_memory().available,
                         free,total,shutil.disk_usage(runtime).free,host.get('free_bytes') if fresh else None):
            raise RuntimeError('BLOCKED_RESOURCE: fresh host/RAM/VRAM/disk admission failed')
        torch.cuda.set_per_process_memory_fraction(min(.75,(free-1024**3)/total))
        torch.cuda.reset_peak_memory_stats()
        resource={'initial_free_vram':free,'total_vram':total,'ram_estimate':ram_estimate,'vram_estimate':vram_estimate,
                  'host_disk_evidence':host,'runtime_free_disk':shutil.disk_usage(runtime).free}
    method_metadata={}
    def fit():
        if family=='rgmf_residual_gru':
            from .residual import ResidualRGMFCUDA
            if raw_sequences is None:raise ValueError('Raw sequences required for leakage-free OOF preprocessing')
            before=time.monotonic();width,lr=trial['neural']
            rx=raw_sequences[training.sequence_index];rv=raw_sequences[testing.sequence_index]
            model=ResidualRGMFCUDA([list(range(2,10))],[10,11],base_columns=[0,1,12,13],kind='gru',width=width,lr=lr,epochs=100,seed=seed)
            model.fit(rx,training.y.to_numpy(),scale,training.decision_time,training.label_end,training.label_available_at,
                      transform.fit_cutoff,np.isfinite(rx));mu,q=model.predict(rv,np.isfinite(rv))
            method_metadata.update({'method':'two_stage_mature_prequential','base':'Ridge_autoregressive_seasonal',
                                    'oof_rows':model.oof_count,'interpretation':'different declared base/method; not an isolated neural-base-freeze effect'})
            return model,mu,q,time.monotonic()-before
        if family=='ssl_gru':
            from .ssl import pretrain_gru
            before=time.monotonic();width,lr=trial['neural']
            pretrain=pretrain_gru(tx,training.decision_time,training.max_dependency_available_at,transform.fit_cutoff,width=width,epochs=100,seed=seed)
            initialization=digest({'pretrain_data_id':pretrain['data_id'],'method':pretrain['method'],'seed':seed,'width':width,'epochs':100})
            model=NeuralCUDA('gru',width=width,lr=lr,epochs=100,seed=seed).fit(tx,training.y.to_numpy(),scale,
                    observed=np.ones_like(tx,dtype=bool),initial_encoder_state=pretrain['encoder_state'],initialization_id=initialization,
                    checkpoint_path=runtime/'checkpoints/auxiliary'/(identity+'.pt'))
            mu,q=model.predict(vx);method_metadata.update({'method':'train_cutoff_SSL_then_supervised','pretrain_data_id':pretrain['data_id'],
                      'pretrain_cutoff':pretrain['cutoff'],'pretrain_epochs':100,'initialization_id':initialization,
                      'independent_release_count_not_multiplied_by_overlapping_contexts':True})
            return model,mu,q,time.monotonic()-before
        return fit_predict(family,tx,training.y.to_numpy(),scale,vx,seed,trial['regularization'],trial['neural'],
                           runtime/'checkpoints/auxiliary'/(identity+'.pt'))
    started=time.monotonic()
    if family in CPU_FAMILIES:model,mean,q,seconds=fit()
    else:
        with budget.charge(identity,reservation):model,mean,q,seconds=fit()
    if not np.isfinite(mean).all() or not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():
        raise RuntimeError('Invalid model forecast')
    metric={'family':family,'seed':seed,'trial':trial,'state':'SUCCEEDED','evidence_kind':'development',
            'n_unique_dates':int(testing.decision_time.nunique()),'pinball':float(quantile_loss(testing.y.to_numpy(),q,QUANTILES,scale).mean()),
            'normalized_mse':float(np.mean(((testing.y.to_numpy()-mean)/scale)**2)),
            'normalizer_id':normalizer,'data_id':data_id,'run_id':identity,'seconds':seconds,
            'full_wall_seconds':time.monotonic()-started,'pit_tier':'B','model_artifact':'model.bin',
            'parameter_count':getattr(model,'parameter_count',None),'scale':scale}
    if family not in CPU_FAMILIES:resource['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    metric['resource_admission']=resource
    metric['feature_variant']=feature_variant;metric['method_metadata']=method_metadata
    if family.startswith('rgmf_'):metric['raw_source_validity_columns']=[list(range(2,10))]
    predictions={'decision_time':[date.isoformat() for date in pd.to_datetime(testing.decision_time,utc=True)],
                 'y':testing.y.tolist(),'mean':mean.tolist(),'quantiles':q.tolist()}
    temp_root=runtime/'tmp';temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as temp:
        model_path=Path(temp)/'model.bin';model.save(model_path)
        commit_bundle(directory,{'metrics.json':metric,'predictions.json':predictions,'transform.json':transform.manifest(),
                                 'model.bin':model_path.read_bytes()}, {'code_hash':code,'data_id':data_id,'run_id':identity})
    print(json.dumps({'family':family,'seed':seed,'run_id':identity,'state':'SUCCEEDED','seconds':seconds,'evidence_kind':'development'}),flush=True)
    return metric


def run_auxiliary(repo,runtime,scope='all',cache_only=False):
    if scope not in {'all','cpu_only'}:raise ValueError('Registered compute scope required')
    if cache_only and getattr(fit_artifact,'cache_only_training_forbidden',False) is not True:
        raise PermissionError('Cache-only admission requires the unconditional no-training guard')
    families=FAMILIES if scope=='all' else [f for f in FAMILIES if f in CPU_FAMILIES]
    acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
    if len(acquisition.get('records',[]))!=acquisition['required']:
        raise RuntimeError('BLOCKED_DATA: complete discovered chronological release accounting required')
    study=json.loads((repo/'configs/study.json').read_text())
    dependency_end=pd.Timestamp(f"{max(study['splits']['development_outer_years'])+1}-01-15",tz='UTC')
    unresolved=[r for r in acquisition['records'] if r['state']!='SUCCEEDED']
    import re
    for record in unresolved:
        match=re.search(r'/([0-9]{4}_[0-9]{2}_[0-9]{2})/',record['release_page'])
        if not match or pd.Timestamp(match[1].replace('_','-'),tz='UTC')<=dependency_end:
            raise RuntimeError('BLOCKED_DATA: unresolved release inside registered development dependency window')
    frame,x=sequences(auxiliary_rows(acquisition['records']))
    import psutil
    host=json.loads((repo/'reports/host_disk_evidence.json').read_text())
    available_ram=psutil.virtual_memory().available
    ram_estimate=max(512*1024**2,x.nbytes*20)
    if (ram_estimate>min(8*1024**3,.6*available_ram) or host['free_bytes']<21*1024**3 or
        pd.Timestamp.now(tz='UTC')-pd.Timestamp(host['observed_at'])>pd.Timedelta(hours=2) or
        shutil.disk_usage(runtime).free<11*1024**3):
        raise RuntimeError('BLOCKED_RESOURCE: panel RAM/host/runtime admission')
    profile=json.loads((repo/'configs/local_rtx4090_laptop.json').read_text())
    config={'study_id':'sgqx-v3','substudy':'auxiliary_physical_only_reconstructed_development','pit_tier':'B','lookback':26,
            'families':families,'registered_full_families':FAMILIES,'compute_scope':scope,
            'diagnostic_reference':'lightgbm' if scope=='all' else 'ridge',
            'outer_years':study['splits']['development_outer_years'],'seeds':study['hpo']['development_seeds'],
            'pilot_max_trials':6,'pilot_measured_folds':2,'pilot_cost_configuration':'largest registered capacity; all three seeds',
            'pilot_completed_fit_ceiling':profile['budget']['pilot_completed_fit_ceiling'],
            'full_trials':20,'trial_grids':{f:trial_grid(f) for f in families},
            'inner_validation_dates':26,'selection':'inner mean multi-quantile normalized pinball; deterministic trial-index ties',
            'epochs':100,'cost_multiplier':2.,'budget_seconds':profile['budget']['development_gpu_seconds'],
            'interpretation':'Tier B physical-only diagnostic; Main/Nested/full-study/final qualification separate',
            'joint_residual_scope':'joint adapter here; two-stage is a separately tested and separately registered method',
            'variants':{'lgb_no_age':'retrained metadata ablation','lgb_no_coverage':'retrained mask ablation; constant masks limit interpretation',
                        'rgmf_residual_gru':'Ridge-base mature-prequential method, not isolated neural-base freezing','ssl_gru':'100 train-cutoff pretraining epochs then same 100 downstream epochs'},
            'final_access':False}
    config['source_gate']={'registered_dependency_end':dependency_end.isoformat(),
                           'unresolved_outside_development_dependencies':unresolved,
                           'full_archive_complete':acquisition['completed']==acquisition['required'],
                           'qualification':'Only fixed 2018-2022 development dependencies admitted; later source failures remain blocked'}
    config['panel_admission']={'ram_estimate_bytes':ram_estimate,'available_ram_bytes':available_ram,
                                'host_disk_evidence':host,'scope':scope}
    atomic_json(repo/'reports/auxiliary_registered_protocol.json',config)
    code=code_hash(repo)
    from .input_identity import canonical_data_id,snapshot_inputs
    data_id=canonical_data_id(acquisition,frame,x)
    folds={};results=[];pilot_costs={}
    for year in config['outer_years']:
        cutoff=pd.Timestamp(f'{year}-01-01T00:00Z')-pd.Timedelta(nanoseconds=1)
        test=frame[(frame.decision_time>=f'{year}-01-01')&(frame.decision_time<f'{year+1}-01-01')]
        train=purged_training(frame,cutoff,list(zip(test.label_start,test.label_end)))
        if train.decision_time.nunique()<156 or test.empty:
            results.append({'outer_year':year,'state':'BLOCKED_DATA','reason':'Insufficient distinct history or outer coverage'});continue
        dates=sorted(train.decision_time.unique())[-26:];valid=train[train.decision_time.isin(dates)]
        inner_cut=pd.Timestamp(dates[0])-pd.Timedelta(nanoseconds=1)
        inner=purged_training(train,inner_cut,list(zip(valid.label_start,valid.label_end)))
        if valid.decision_time.nunique()<13 or inner.decision_time.nunique()<52:
            results.append({'outer_year':year,'state':'BLOCKED_DATA','reason':'Insufficient inner temporal history'});continue
        folds[year]={'train':train,'test':test,'inner':inner,'valid':valid,
                     'inner_arrays':preprocess(x,inner,valid,inner_cut),'outer_arrays':preprocess(x,train,test,cutoff)}
    if not folds:raise RuntimeError('BLOCKED_DATA: no admissible chronological folds')
    budget=ComputeBudget(runtime/'ledger/development_compute.sqlite',config['budget_seconds'])
    trials=TrialLedger(runtime/'ledger/auxiliary_trials.sqlite')
    from .stage_cache import record_training_dependencies,legacy_compatible,adopt_bundle
    training_id=record_training_dependencies(repo,runtime)
    snapshot_inputs(runtime,acquisition,frame,x,training_id)
    from .correction_cache import corrected_cache,adopt_corrected
    correction_parents,correction_proof=corrected_cache(repo,runtime,folds,x,data_id)
    atomic_json(repo/'reports/source_correction_cache_audit.json',{'proof_id':correction_proof,
                'exact_input_reusable_bundles':len(correction_parents),'data_id':data_id,'reserved_access':False})
    legacy_inner={}
    for old_id,serialized,state,_,_ in trials.rows():
        old=json.loads(serialized);original_code=old.pop('code',None)
        if state=='SUCCEEDED' and original_code and legacy_compatible(runtime,original_code,training_id):
            legacy_inner.setdefault(digest(old),old_id)
    legacy_outer={}
    snapshots=runtime/'artifacts/auxiliary_runs'
    for directory in snapshots.iterdir() if snapshots.exists() else []:
        if not (directory/'receipt.json').exists():continue
        validate_bundle(directory)
        for row in json.loads((directory/'results.json').read_text()).get('results',[]):
            if row.get('state')!='SUCCEEDED' or not row.get('run_id'):continue
            key=digest({k:row[k] for k in ['outer_year','family','seed','trial','normalizer_id','data_id']})
            legacy_outer.setdefault(key,row['run_id'])
    rows=[]
    def hpo(family,year,index,seed=11):
        fold=folds[year];trial=trial_grid(family)[index]
        configuration={'partition':'inner_validation','experiment':'E02','family':family,'year':year,'seed':seed,
                       'trial':trial,'data_id':data_id,'training_implementation_id':training_id,'substudy':config['substudy']}
        identity=trials.register(configuration);prior=trials.get(identity)
        original_identity=identity
        if (prior[2] in {'PLANNED','RUNNING'} and
            not (runtime/'artifacts/auxiliary'/identity/'receipt.json').exists() and
            budget.db.execute('SELECT id FROM compute_charges WHERE id=?',(identity,)).fetchone()):
            trials.finish(identity,'INTERRUPTED',failure='Prior charged attempt has no completed artifact')
            prior=trials.get(identity)
        if prior[2] in {'FAILED_RETRYABLE','INTERRUPTED','PAUSED_BUDGET'}:
            try:identity=trials.bounded_retry(original_identity,maximum_attempts=2);prior=trials.get(identity)
            except RuntimeError as exc:
                rows.append({'outer_year':year,'family':family,'trial_index':index,'state':'BLOCKED_RECOVERY','reason':str(exc)})
                return None
        if prior[2]=='SUCCEEDED':
            validate_bundle(runtime/'artifacts/auxiliary'/identity)
            return json.loads((runtime/'artifacts/auxiliary'/identity/'metrics.json').read_text())
        if prior[2] not in {'PLANNED','RUNNING'}:return None
        try:
            correction_key=digest({'partition':'inner_validation','year':year,'family':family,'seed':seed,'trial':trial})
            corrected=adopt_corrected(runtime,correction_parents.get(correction_key),identity,code,training_id,
                       {'family':family,'trial':trial,'seed':seed,'data_id':data_id,'normalizer_id':fold['inner_arrays'][3]},
                       correction_proof,'inner_validation',year)
            if corrected:
                trials.finish(identity,'SUCCEEDED',corrected['pinball']);return corrected
            legacy_key=digest({k:v for k,v in configuration.items() if k!='training_implementation_id'})
            parent=legacy_inner.get(legacy_key)
            if parent:
                reused=adopt_bundle(runtime,parent,identity,code,training_id,
                       {'family':family,'trial':trial,'seed':seed,'data_id':data_id,'normalizer_id':fold['inner_arrays'][3]})
                if reused:
                    trials.finish(identity,'SUCCEEDED',reused['pinball']);return reused
            if family not in CPU_FAMILIES:
                from .runtime import ensure_gpu_owner
                ensure_gpu_owner()
            if identity!=original_identity:
                from .runtime import atomic_bytes
                original=runtime/'checkpoints/auxiliary'/(original_identity+'.pt')
                destination=runtime/'checkpoints/auxiliary'/(identity+'.pt')
                receipt=original.with_suffix('.receipt.json')
                if original.exists() and receipt.exists() and not destination.exists():
                    from .runtime import file_hash
                    if file_hash(original)!=json.loads(receipt.read_text())['sha256']:
                        raise RuntimeError('BLOCKED_RECOVERY: retry parent checkpoint corrupt')
                    atomic_bytes(destination,original.read_bytes())
                    atomic_bytes(destination.with_suffix('.receipt.json'),receipt.read_bytes())
            metric=fit_artifact(repo,runtime,budget,family,trial,seed,fold['inner'],fold['valid'],fold['inner_arrays'],identity,code,data_id,raw_sequences=x)
            trials.finish(identity,'SUCCEEDED',metric['pinball']);return metric
        except Exception as exc:
            if 'BLOCKED_GPU' in str(exc):
                trials.finish(identity,'FAILED_RETRYABLE',failure='ExternalGPUOwner');raise
            state='PAUSED_BUDGET' if 'PAUSED_BUDGET' in str(exc) else 'INTERRUPTED' if 'already charged' in str(exc) else 'FAILED_PERMANENT'
            trials.finish(identity,state,failure=type(exc).__name__)
            rows.append({'outer_year':year,'family':family,'trial_index':index,'state':state,'reason':str(exc)})
            return None
    try:
        with gpu_lease(runtime) if scope=='all' and not cache_only else nullcontext():
            pilot_years=sorted(folds)[:2]
            if len(pilot_years)<2:raise RuntimeError('BLOCKED_DATA: two complete pilot development folds required')
            planned_pilot_count=len(pilot_years)*len(families)*len(config['seeds'])
            if planned_pilot_count>config['pilot_completed_fit_ceiling']:raise RuntimeError('PAUSED_BUDGET: pilot fit ceiling')
            for family in families:
                measurements=[]
                grid=trial_grid(family)
                for year in pilot_years:
                    for seed in config['seeds']:
                        metric=hpo(family,year,len(grid)-1,seed)
                        if metric:measurements.append(metric['seconds'])
                pilot_costs[family]={'successful_pilot_fits':len(measurements),'max_fit_seconds':max(measurements) if measurements else None,
                                     'measured_years':pilot_years,'seeds':config['seeds'],'capacity_trial_index':len(grid)-1}
                atomic_json(repo/'reports/auxiliary_pilot_costs.json',{'data_id':data_id,'code_hash':code,'families':pilot_costs,'remaining_gpu_seconds':budget.remaining})
            counts={f:len(folds)*(len(trial_grid(f))+len(config['seeds']))+len(pilot_years)*(len(config['seeds'])-1) for f in families}
            estimated=sum(pilot_costs[f]['max_fit_seconds']*counts[f]*config['cost_multiplier'] for f in families
                          if f not in CPU_FAMILIES and pilot_costs[f]['max_fit_seconds'] is not None)
            from .resume_admission import remaining_fit_counts
            remaining_counts,resume_evidence=remaining_fit_counts(runtime,families,list(folds),config['seeds'],
                {f:trial_grid(f) for f in families},correction_parents)
            full_grid_estimate=estimated
            estimated=sum(pilot_costs[f]['max_fit_seconds']*remaining_counts[f]*config['cost_multiplier'] for f in families
                          if f not in CPU_FAMILIES and pilot_costs[f]['max_fit_seconds'] is not None)
            if cache_only:
                if getattr(fit_artifact,'cache_only_training_forbidden',False) is not True:
                    raise PermissionError('Cache-only admission requires the unconditional no-training guard')
                qualification=json.loads((repo/'reports/canonical_cache_upgrade_qualification.json').read_text())
                if (qualification.get('canonical_data_id')!=data_id or qualification.get('missing_recipe_ids') or
                    not qualification.get('all_registered_recipes_required') or
                    qualification['all_registered_recipes_required']!=qualification.get('all_registered_recipes_reusable')):
                    raise RuntimeError('BLOCKED_CACHE_PROOF: complete exact-input replay proof required')
                estimated=0.
            bill={'data_id':data_id,'code_hash':code,'pilot_costs':pilot_costs,'planned_fit_counts':counts,
                  'unmeasured_families':[f for f in families if pilot_costs[f]['max_fit_seconds'] is None],
                  'expected_remaining_gpu_seconds_conservative':estimated,'remaining_gpu_seconds':budget.remaining,
                  'state':'READY' if estimated<=budget.remaining else 'PAUSED_BUDGET','created_at':now(),
                  'full_trials':20,'physical_gpus':1,'reserved_access':False}
            bill['admitted_work']='verified_cache_only_materialization' if cache_only else 'full_development'
            bill['full_grid_gpu_seconds_conservative']=full_grid_estimate
            bill['remaining_fit_counts']=remaining_counts
            bill['exact_input_resume_evidence']=resume_evidence
            bill['expected_cpu_fit_seconds_conservative']=sum(pilot_costs[f]['max_fit_seconds']*counts[f]*config['cost_multiplier'] for f in CPU_FAMILIES
                                                              if pilot_costs[f]['max_fit_seconds'] is not None)
            if cache_only:bill['expected_cpu_fit_seconds_conservative']=0.
            atomic_json(repo/'reports/auxiliary_measured_run_bill.json',bill)
            if bill['state']!='READY':
                atomic_json(repo/'reports/auxiliary_development_results.json',{'protocol':config,'results':results+rows,'state':'PAUSED_BUDGET','bill':bill})
                return results+rows
            for year,fold in folds.items():
                for family in families:
                    if pilot_costs[family]['max_fit_seconds'] is None:
                        results.append({'outer_year':year,'family':family,'state':'FAILED_PERMANENT',
                                        'reason':'No successful measured pilot; full fitting not resource-admitted; registered fallback retained'})
                        continue
                    candidates=[]
                    for index in range(len(trial_grid(family))):
                        metric=hpo(family,year,index)
                        if metric:candidates.append((metric['pinball'],index))
                    if not candidates:
                        results.append({'outer_year':year,'family':family,'state':'FAILED_PERMANENT','reason':'All inner trials failed'});continue
                    chosen=min(candidates)[1];trial=trial_grid(family)[chosen]
                    for seed in config['seeds']:
                        identity=digest({'partition':'outer_development','family':family,'year':year,'seed':seed,'trial':trial,
                                         'data_id':data_id,'training_implementation_id':training_id,'normalizer':fold['outer_arrays'][3]})
                        try:
                            correction_key=digest({'partition':'outer_development','year':year,'family':family,'seed':seed,'trial':trial})
                            metric=adopt_corrected(runtime,correction_parents.get(correction_key),identity,code,training_id,
                                   {'family':family,'trial':trial,'seed':seed,'data_id':data_id,'normalizer_id':fold['outer_arrays'][3]},
                                   correction_proof,'outer_development',year)
                            key=digest({'outer_year':year,'family':family,'seed':seed,'trial':trial,
                                        'normalizer_id':fold['outer_arrays'][3],'data_id':data_id})
                            parent=legacy_outer.get(key)
                            metric=metric or (adopt_bundle(runtime,parent,identity,code,training_id,
                                   {'family':family,'trial':trial,'seed':seed,'data_id':data_id,'normalizer_id':fold['outer_arrays'][3]}) if parent else None
                                   )
                            if metric is None:
                                if family not in CPU_FAMILIES:
                                    from .runtime import ensure_gpu_owner
                                    ensure_gpu_owner()
                                metric=fit_artifact(repo,runtime,budget,family,trial,seed,fold['train'],fold['test'],fold['outer_arrays'],identity,code,data_id,raw_sequences=x)
                            results.append({**metric,'outer_year':year,'selected_trial_index':chosen})
                        except Exception as exc:
                            if 'BLOCKED_GPU' in str(exc):raise
                            results.append({'outer_year':year,'family':family,'seed':seed,'run_id':identity,
                                            'state':'PAUSED_BUDGET' if 'PAUSED_BUDGET' in str(exc) else 'FAILED_PERMANENT','reason':str(exc)})
                        atomic_json(repo/'reports/auxiliary_development_results.json',{'protocol':config,'results':results+rows,
                                     'remaining_gpu_seconds':budget.remaining,'state':'RUNNING','authoritative_full_study':False})
    finally:
        budget.close();trials.close()
    atomic_json(repo/'reports/auxiliary_development_results.json',{'protocol':config,'results':results+rows,
                 'state':'SUCCEEDED_DIAGNOSTIC' if all(r['state']=='SUCCEEDED' for r in results) else 'PARTIAL',
                 'authoritative_full_study':False,'required_comparisons_unfinished':True})
    commit_bundle(runtime/'artifacts/auxiliary_runs'/digest({'config':config,'data_id':data_id,'code':code}),
                  {'results.json':json.loads((repo/'reports/auxiliary_development_results.json').read_text())},
                  {'data_id':data_id,'code_hash':code,'compute_scope':scope})
    from .analysis import analyze_auxiliary
    analyze_auxiliary(repo,runtime)
    return results+rows
