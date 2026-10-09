"""Chronological multi-asset development engine; source qualification is separate.

Input panels must be built from registered release-aware sources. This engine
never acquires prices, reads reserved targets, or treats fixture fits as research.
"""
import json
import tempfile
import io
from pathlib import Path
import numpy as np
import pandas as pd
from .features import TrainTransform
from .panels import matched_grid,track_folds
from .models import Statistical,CudaTree,QUANTILES
from .track_statistics import DateWeightedStatistical
from .metrics import quantile_loss
from .runtime import digest,commit_bundle,validate_bundle,file_hash,gpu_lease,atomic_json

CPU_FAMILIES={'historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage'}
NEURAL_FAMILIES={'mlp','gru','gru_d','tft','transformer','rgmf_linear','rgmf_gru','rgmf_transformer','ssl_gru','rgmf_residual_gru'}


def track_operator_id(repo,study,track):
    names=['track_engine','models','features','pit','neural','track_neural','ssl','track_ssl','residual','track_residual','track_statistics']
    environment=json.loads((repo/'reports/environment_audit.json').read_text())
    return digest({'math_files':{name:file_hash(repo/'src/signalforge'/(name+'.py')) for name in names},
        'folds':study['splits']['track_folds'][track],'hpo':study['hpo'],
        'environment':{name:environment[name] for name in ['resolved_packages','python','dispatcher_sha256']}})


def validate_panel(panel,contexts):
    # Check the date boundary before inspecting numerical targets.
    if any(pd.Timestamp(t).tzinfo is None for t in panel.decision_time):raise ValueError('Aware origins required')
    dates=pd.to_datetime(panel.decision_time,utc=True,format='mixed')
    if dates.ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved targets forbidden in development')
    matched_grid(panel)
    required={'y','label_start','label_end','label_available_at','max_dependency_available_at','sequence_index','context_eligible','eligible'}
    if not required<=set(panel):raise ValueError('Complete target/context lineage required')
    x=np.asarray(contexts,dtype=float)
    if x.ndim!=3 or len(x)!=len(panel) or np.isinf(x).any():raise ValueError('Finite-or-missing aligned contexts required')
    if not np.array_equal(np.sort(panel.sequence_index),np.arange(len(panel))):raise ValueError('Unique sequence indices required')
    for column in ['label_start','label_end','label_available_at','max_dependency_available_at']:
        if any(pd.Timestamp(t).tzinfo is None for t in panel[column].dropna()):raise ValueError('Aware target/dependency clocks required')
    if panel.max_dependency_available_at.isna().any():raise ValueError('Complete feature dependency clocks required')
    if panel.loc[pd.to_numeric(panel.y).notna(),['label_start','label_end','label_available_at']].isna().any().any():raise ValueError('Observed target requires complete maturity clocks')
    if (pd.to_datetime(panel.max_dependency_available_at,utc=True)>dates).any():raise ValueError('Future feature dependency')
    start=pd.to_datetime(panel.label_start,utc=True);end=pd.to_datetime(panel.label_end,utc=True)
    available=pd.to_datetime(panel.label_available_at,utc=True)
    p0=panel.get('price_mode',pd.Series('P1',index=panel.index)).eq('P0')
    if ((start<dates)&~p0).any() or (end<=start).any() or (available<end).any():raise ValueError('Invalid actual target intervals/publication')
    if p0.any():
        if 'p0_origin_mark_available_at' not in panel:raise ValueError('P0 origin-mark publication required')
        known=pd.to_datetime(panel.p0_origin_mark_available_at,utc=True,format='mixed')
        observed=p0&pd.to_numeric(panel.y).notna()
        if (known[observed].isna().any() or (known[observed]>dates[observed]).any() or (start[p0]>dates[p0]).any() or (end[p0]<=dates[p0]).any()):
            raise ValueError('P0 mark must be known at origin and target must follow it')
    if np.isinf(pd.to_numeric(panel.y)).any():raise ValueError('Infinite target')
    if 'price_mode' in panel and (panel.price_mode.nunique()!=1 or not panel.price_mode.isin(['P0','P1']).all()):raise ValueError('Separate explicit price protocols required')
    return x


def asset_normalizer(training,cutoff):
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or training.empty:raise ValueError('Aware nonempty training contract required')
    if (pd.to_datetime(training.label_available_at,utc=True)>cutoff).any():raise ValueError('Immature normalization targets')
    scales={}
    for asset,rows in training.groupby('asset',sort=True):
        y=rows.y.to_numpy(dtype=float)
        if not np.isfinite(y).all():raise ValueError('Normalizer requires mature finite targets')
        scales[asset]=max(float(y.std()),1e-8)
    lineage=training[['asset','decision_time','y','label_available_at']].copy()
    for column in ['decision_time','label_available_at']:lineage[column]=pd.to_datetime(lineage[column],utc=True,format='mixed').astype(str)
    identity=digest({'cutoff':cutoff.isoformat(),'rows':lineage.sort_values(['decision_time','asset']).to_dict('records'),'scales':scales})
    return scales,identity


def per_date_weights(frame):
    # Each market date has equal weight; assets share their date's mass.
    counts=frame.groupby('decision_time').asset.transform('count').to_numpy()
    return 1./counts


def prepare(training,testing,x,cutoff):
    training=training.sort_values(['decision_time','asset'])
    tx=x[training.sequence_index];vx=x[testing.sequence_index]
    transform=TrainTransform().fit(tx.reshape(-1,tx.shape[-1]),np.repeat(training.decision_time.to_numpy(),tx.shape[1]),cutoff)
    def apply(a):return transform.transform(a.reshape(-1,a.shape[-1])).reshape(len(a),-1)
    scales,normalizer_id=asset_normalizer(training,cutoff)
    if not set(testing.asset)<=set(scales):raise ValueError('Test asset absent from common mature training grid')
    train_scale=training.asset.map(scales).to_numpy();test_scale=testing.asset.map(scales).to_numpy()
    lineage=training[[c for c in ['decision_time','asset','max_dependency_available_at','feature_lineage','context_lineage','raw_hash','target_raw_hash','target_raw_hashes','price_mode','p0_basis','label_start','label_end','label_available_at'] if c in training]].copy()
    for column in ['decision_time','max_dependency_available_at','label_start','label_end','label_available_at']:
        if column in lineage:lineage[column]=pd.to_datetime(lineage[column],utc=True,format='mixed').astype(str)
    ft,fv=apply(tx),apply(vx)
    observed=np.isfinite(tx);test_observed=np.isfinite(vx)
    result={'training':training,'tx':ft,'vx':fv,'y':training.y.to_numpy()/train_scale,'training_lineage':lineage.to_dict('records'),
            'test_scale':test_scale,'asset_scales':scales,'normalizer_id':normalizer_id,'transform':transform,
            'weights':per_date_weights(training),'tx_sequence':ft.reshape(len(tx),tx.shape[1],-1),'vx_sequence':fv.reshape(len(vx),vx.shape[1],-1),
            'observed':np.concatenate([observed,np.ones_like(observed)],axis=-1),
            'test_observed':np.concatenate([test_observed,np.ones_like(test_observed)],axis=-1)}
    result.update(raw_tx=tx,raw_vx=vx)
    if 'context_lineage' in training and 'context_lineage' in testing:
        from .track_neural import elapsed_from_context_clocks
        def clocks(frame):
            return [[None]*(x.shape[1]-len(rows))+[r['decision_time'] for r in rows] for rows in frame.context_lineage]
        result['elapsed']=elapsed_from_context_clocks(tx,clocks(training),training.decision_time.tolist())
        result['test_elapsed']=elapsed_from_context_clocks(vx,clocks(testing),testing.decision_time.tolist())
    if 'neural_contract' in training.attrs:
        from .track_neural import raw_source_validity
        contract=training.attrs['neural_contract'];dimension=tx.shape[-1]
        def expanded(columns):return columns+[column+dimension for column in columns]
        result['neural_args']={'base_columns':expanded(contract['base_raw_columns']),
            'source_columns':[expanded(columns) for columns in contract['source_raw_columns']],
            'meta_columns':expanded(contract['meta_raw_columns'])}
        result['raw_source_value_columns']=contract['source_value_columns']
        result['raw_neural_contract']=contract
        result['source_valid']=raw_source_validity(tx,contract['source_value_columns'])
        result['test_source_valid']=raw_source_validity(vx,contract['source_value_columns'])
    return result


def score(testing,mean,q,scales):
    eligible=testing.eligible.to_numpy(dtype=bool)&np.isfinite(testing.y.to_numpy(dtype=float))
    if not np.isfinite(mean).all() or not np.isfinite(q).all():raise ValueError('Missing predictions must use registered fallback')
    rows=testing[['decision_time','asset']].copy()
    rows['decision_time']=pd.to_datetime(rows.decision_time,utc=True).astype(str)
    rows['scorable']=eligible;rows['pinball']=np.nan;rows['normalized_mse']=np.nan
    rows.loc[eligible,'pinball']=quantile_loss(testing.y.to_numpy()[eligible],q[eligible],QUANTILES,scales[eligible])
    rows.loc[eligible,'normalized_mse']=((testing.y.to_numpy()[eligible]-mean[eligible])/scales[eligible])**2
    date=rows.groupby('decision_time')[['pinball','normalized_mse']].mean()
    return {'pinball':float(date.pinball.mean()) if eligible.any() else None,'normalized_mse':float(date.normalized_mse.mean()) if eligible.any() else None,
            'n_grid_rows':len(rows),'n_scorable_rows':int(eligible.sum()),'n_grid_dates':int(rows.decision_time.nunique()),
            'n_scored_dates':int(date.pinball.notna().sum()),'date_losses':date.reset_index().astype(object).where(date.reset_index().notna(),None).to_dict('records')}


def run_panel_development(repo,runtime,panels,contexts,study,track,families,budget=None,evidence_kind='synthetic_fixture'):
    """Full fixed inner grid and three outer seeds, shared information comparisons.

    CUDA trees require the caller's actual budget and acquire the project lease.
    Main/Nested cannot be scientifically qualified by this software execution.
    """
    if track not in study['splits']['track_folds']:raise ValueError('Registered track required')
    if not families or len(families)!=len(set(families)):raise ValueError('Unique fixed families required')
    if not set(families)<=CPU_FAMILIES|{'lightgbm','xgboost'}|NEURAL_FAMILIES:raise ValueError('Unsupported registered track family')
    if set(families)-CPU_FAMILIES and budget is None:raise PermissionError('Measured CUDA budget required')
    if set(panels)!=set(contexts) or not panels:raise ValueError('Aligned information sets required')
    grid_id=matched_grid(*panels.values());folds={};blocked=[];arrays={}
    for info,panel in panels.items():
        arrays[info]=validate_panel(panel,contexts[info]);folds[info],fail=track_folds(panel,study,track);blocked.extend(fail)
    # Common mature/context-eligible train rows, never per-model favorable subsets.
    years=set.intersection(*(set(f) for f in folds.values()))
    result=[];tree_code=track_operator_id(repo,study,track)
    from contextlib import nullcontext
    lease=gpu_lease(runtime) if set(families)-CPU_FAMILIES else nullcontext()
    with lease:
        for year in sorted(years):
            common={}
            for partition in ['inner','train']:
                keys=[set(zip(folds[info][year][partition].loc[folds[info][year][partition].context_eligible,'decision_time'],folds[info][year][partition].loc[folds[info][year][partition].context_eligible,'asset'])) for info in panels]
                common[partition]=set.intersection(*keys)
                if not common[partition]:raise ValueError('No common eligible training rows')
            shared_ids={}
            for info in sorted(panels):
                fold=folds[info][year];prepared={}
                for partition,testpart,cutoff in [('inner','valid','inner_cutoff'),('train','test','outer_cutoff')]:
                    train=fold[partition]
                    train=train[[key in common[partition] for key in zip(train.decision_time,train.asset)]]
                    prepared[partition]=prepare(train,fold[testpart],arrays[info],fold[cutoff])
                    previous=shared_ids.setdefault(partition,prepared[partition]['normalizer_id'])
                    if previous!=prepared[partition]['normalizer_id']:raise ValueError('Information sets disagree on shared target rows/scales')
                for family in families:
                    trials=([1.] if family in {'historical','ewma'} else
                        [[width,lr] for width in [8,16,32,48,64] for lr in [.0003,.001,.003,.01]] if family in NEURAL_FAMILIES else
                        np.geomspace(.0001,10000,study['hpo']['full_trials_per_family_fold']).tolist())
                    inner=[]
                    for regularization in trials:
                        fit=_fit_registered(repo,runtime,prepared['inner'],fold['valid'],family,regularization,study['hpo']['development_seeds'][0],track,info,year,'inner',tree_code,grid_id,budget,evidence_kind)
                        inner.append(fit)
                    successful=[fit for fit in inner if fit['state']=='SUCCEEDED' and fit['pinball'] is not None and np.isfinite(fit['pinball'])]
                    if not successful:
                        blocked.append({'track':track,'information':info,'year':year,'family':family,'state':'FAILED_ALL_INNER','attempts':inner});continue
                    winner=min(successful,key=lambda fit:(fit['pinball'],fit['regularization']))
                    for seed in study['hpo']['development_seeds']:
                        fit=_fit_registered(repo,runtime,prepared['train'],fold['test'],family,winner['regularization'],seed,track,info,year,'outer',tree_code,grid_id,budget,evidence_kind)
                        result.append({**fit,'inner_selection_id':winner['run_id'],'inner_trial_count':len(inner),'inner_attempts':inner})
    complete=not blocked and len(result)==len(study['splits']['track_folds'][track]['outer_years'])*len(panels)*len(families)*len(study['hpo']['development_seeds']) and all(r['state']=='SUCCEEDED' for r in result)
    return {'state':'SUCCEEDED_DEVELOPMENT_SOFTWARE' if complete else 'PARTIAL' if result else 'BLOCKED_DATA','track':track,'results':result,'blocked_folds':blocked,
            'evidence_kind':evidence_kind,'qualified_for_final':False,'economic_qualified':False,'reserved_access':False,'grid_id':grid_id}


def _fit_registered(repo,runtime,prepared,testing,family,regularization,seed,track,info,year,stage,code,grid_id,budget,evidence_kind):
    recipe={'track':track,'information':info,'year':year,'stage':stage,'family':family,'regularization':regularization,'seed':seed,
            'code':code,'grid_id':grid_id,'normalizer_id':prepared['normalizer_id'],'transform':prepared['transform'].manifest(),
            'tx':prepared['tx'].tolist(),'y':prepared['y'].tolist(),'weights':prepared['weights'].tolist(),'evidence_kind':evidence_kind}
    recipe['training_lineage']=prepared['training_lineage']
    recipe['price_modes']=sorted(testing.get('price_mode',pd.Series('P1',index=testing.index)).unique().tolist())
    if family in NEURAL_FAMILIES:
        recipe.update(sequence_shape=list(prepared['tx_sequence'].shape),observed=prepared['observed'].tolist(),
            source_valid=prepared.get('source_valid',[]).tolist() if isinstance(prepared.get('source_valid'),np.ndarray) else None,
            neural_args=prepared.get('neural_args'),raw_source_value_columns=prepared.get('raw_source_value_columns'))
        if family=='gru_d':
            if 'elapsed' not in prepared or 'test_elapsed' not in prepared:raise ValueError('GRU-D actual context clocks required')
            recipe['elapsed_days']=prepared['elapsed'].tolist()
    model_id=digest(recipe);model_directory=runtime/'artifacts/track_models'/model_id
    evaluation={'model_id':model_id,'vx':prepared['vx'].tolist(),'test_labels':testing.y.astype(object).where(testing.y.notna(),None).tolist(),
                'keys':testing[['decision_time','asset']].astype(str).to_dict('records'),'scale':prepared['test_scale'].tolist(),
                'eligible':testing.eligible.tolist(),'context_eligible':testing.context_eligible.tolist()}
    if family in NEURAL_FAMILIES:evaluation.update(test_observed=prepared['test_observed'].tolist(),
        test_source_valid=prepared['test_source_valid'].tolist() if 'test_source_valid' in prepared else None)
    if family=='gru_d':evaluation['test_elapsed_days']=prepared['test_elapsed'].tolist()
    identity=digest(evaluation);destination=runtime/'artifacts/track_development'/identity
    if (destination/'receipt.json').exists():
        validate_bundle(destination);return json.loads((destination/'metrics.json').read_text())
    failure_path=runtime/'artifacts/track_failures'/(identity+'.json')
    if failure_path.exists():return json.loads(failure_path.read_text())
    if family in CPU_FAMILIES:model=DateWeightedStatistical(family,regularization)
    elif family in NEURAL_FAMILIES:
        from .track_neural import ExplicitSourceCUDA
        if family.startswith('rgmf_') and 'neural_args' not in prepared:raise ValueError('Registered raw source groups required for RGMF')
        width,lr=regularization
        if family=='rgmf_residual_gru':
            from .track_residual import TrackResidualCUDA
            model=TrackResidualCUDA(prepared['raw_neural_contract'],width=width,lr=lr,epochs=100,seed=seed)
        else:model=ExplicitSourceCUDA('gru' if family=='ssl_gru' else family,width=width,lr=lr,epochs=100,seed=seed,**(prepared['neural_args'] if family.startswith('rgmf_') else {}))
    else:model=CudaTree(family,rounds=100,seed=seed,regularization=regularization)
    from contextlib import nullcontext
    try:
        if family not in CPU_FAMILIES:
            from .resources import require_current_resources
            require_current_resources(repo,runtime,max(512*1024**2,prepared['tx'].nbytes*20),max(1024**3,prepared['tx'].nbytes*30))
        if (model_directory/'receipt.json').exists():
            model_receipt=validate_bundle(model_directory)
            model=(TrackResidualCUDA.load(model_directory/'model.bin') if family=='rgmf_residual_gru' else Statistical.load(model_directory/'model.bin') if family in CPU_FAMILIES else
                ExplicitSourceCUDA.load(model_directory/'model.bin') if family in NEURAL_FAMILIES else CudaTree.load(model_directory/'model.bin'))
            fallback=Statistical.load(model_directory/'fallback.bin')
        else:
            charge=budget.charge(model_id,300) if budget is not None else nullcontext()
            with charge:
                if family=='rgmf_residual_gru':
                    model.fit(prepared['raw_tx'],prepared['training'],prepared['asset_scales'],prepared['transform'].fit_cutoff,
                        checkpoint_path=runtime/'checkpoints/tracks'/(model_id+'.pt'))
                elif family in NEURAL_FAMILIES:
                    initialization={}
                    if family=='ssl_gru':
                        from .track_ssl import frozen_ssl_initialization
                        state,initialization_id,pretraining=frozen_ssl_initialization(repo,runtime,prepared,width,seed)
                        initialization={'initial_encoder_state':state,'initialization_id':initialization_id}
                    model.fit(prepared['tx_sequence'],prepared['y'],1.,weights=prepared['weights'],observed=prepared['observed'],
                        elapsed=prepared['elapsed'] if family=='gru_d' else np.zeros_like(prepared['tx_sequence']),source_valid=prepared.get('source_valid'),
                        checkpoint_path=runtime/'checkpoints/tracks'/(model_id+'.pt'),**initialization)
                elif family in CPU_FAMILIES:model.fit(prepared['tx'],prepared['y'],weights=prepared['weights'],dates=prepared['training'].decision_time)
                else:model.fit(prepared['tx'],prepared['y'],weights=prepared['weights'])
            fallback=Statistical('historical').fit(prepared['tx'],prepared['y'],weights=prepared['weights'])
            with tempfile.TemporaryDirectory(dir=runtime) as temporary:
                folder=Path(temporary);model.save(folder/'model.bin');fallback.save(folder/'fallback.bin')
                model_receipt=commit_bundle(model_directory,{'model.bin':(folder/'model.bin').read_bytes(),'fallback.bin':(folder/'fallback.bin').read_bytes(),
                    'transform.json':prepared['transform'].manifest(),'asset_scales.json':prepared['asset_scales'],
                    'metrics.json':{'family':family,'bundle_schema':'multi_asset_residual_v1' if family=='rgmf_residual_gru' else 'multi_asset_neural_v1' if family in NEURAL_FAMILIES else 'multi_asset_normalized_v1',
                        'normalizer_id':prepared['normalizer_id'],'elapsed_contract':'actual_context_days_v1' if family=='gru_d' else None,
                        'raw_source_validity_columns':prepared.get('raw_source_value_columns') if family.startswith('rgmf_') else None}},
                    {'recipe_id':model_id,'training_operator_id':code,'evidence_kind':evidence_kind,'normalizer_id':prepared['normalizer_id']})
        mean,q=(model.predict(prepared['raw_vx']) if family=='rgmf_residual_gru' else model.predict(prepared['vx_sequence'],observed=prepared['test_observed'],elapsed=prepared['test_elapsed'] if family=='gru_d' else np.zeros_like(prepared['vx_sequence']),
            source_valid=prepared.get('test_source_valid')) if family in NEURAL_FAMILIES else model.predict(prepared['vx']))
        if mean.shape!=(len(testing),) or q.shape!=(len(testing),5) or not np.isfinite(mean).all() or not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():
            raise ValueError('Invalid registered model predictions')
    except Exception as error:
        if str(error).startswith(('PAUSED_BUDGET','BLOCKED_RESOURCE','BLOCKED_GPU')):raise
        failed={'state':'FAILED_PERMANENT','run_id':identity,'model_id':model_id,'family':family,'seed':seed,'regularization':regularization,
                'failure_type':type(error).__name__,'failure':str(error),'evidence_kind':evidence_kind,'native_prediction_coverage':0.,
                'fallback_policy':'historical_training_distribution','fallback_rows':len(testing)}
        fallback=Statistical('historical').fit(prepared['tx'],prepared['y'],weights=prepared['weights'])
        mean,q=fallback.predict(prepared['vx']);mean*=prepared['test_scale'];q*=prepared['test_scale'][:,None]
        failed.update(score(testing,mean,q,prepared['test_scale']))
        stream=io.BytesIO();np.savez(stream,mean=mean,quantiles=q,model_failure=np.ones(len(testing),dtype=bool))
        commit_bundle(destination,{'metrics.json':failed,'predictions.npz':stream.getvalue()},
            {'recipe_id':identity,'training_operator_id':code,'evidence_kind':evidence_kind,'normalizer_id':prepared['normalizer_id'],'model_failed':True})
        atomic_json(failure_path,failed);return failed
    # Short contexts receive the predeclared historical fallback; no row deletion.
    missing=~testing.context_eligible.to_numpy(dtype=bool)
    fm,fq=fallback.predict(prepared['vx'][missing]);mean[missing]=fm;q[missing]=fq
    mean=mean*prepared['test_scale'];q=q*prepared['test_scale'][:,None]
    metric={**score(testing,mean,q,prepared['test_scale']),'run_id':identity,'family':family,'seed':seed,'regularization':regularization,
            'year':year,'information':info,'track':track,'state':'SUCCEEDED','normalizer_id':prepared['normalizer_id'],
            'context_fallback_rows':int(missing.sum()),'evidence_kind':evidence_kind,'bundle_schema':'multi_asset_normalized_v1',
            'model_id':model_id,'model_relative_bundle':str(model_directory.relative_to(runtime)),'model_receipt_id':digest(model_receipt)}
    stream=io.BytesIO();np.savez(stream,mean=mean,quantiles=q,context_fallback=missing)
    commit_bundle(destination,{'metrics.json':metric,'predictions.npz':stream.getvalue()},
                  {'recipe_id':identity,'training_operator_id':code,'evidence_kind':evidence_kind,'normalizer_id':prepared['normalizer_id']})
    return metric
