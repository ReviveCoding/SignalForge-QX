"""Registered CPU-only 2023 selection/calibration diagnostics; final stays sealed."""
import json,time,tempfile,sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from signalforge.runtime import paths,atomic_json,commit_bundle,validate_bundle,digest,code_hash,now,gpu_lease
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import fit_predict
from signalforge.development import preprocess,CPU_FAMILIES
from signalforge.features import purged_training
from signalforge.stage_cache import record_training_dependencies
from signalforge.stage_cache import training_dependencies
from signalforge.correction_cache import numeric_dependencies
from signalforge.inference import infer_bundle
from signalforge.input_identity import canonical_data_id
from signalforge.ensemble import FrozenEnsemble
from signalforge.frozen_processing import candidate_ensemble_id
from signalforge.postprocess import Calibrator
from signalforge.metrics import quantile_loss
from signalforge.models import QUANTILES

repo,runtime=paths();started=time.monotonic()
all_families='--all' in sys.argv
prefix='auxiliary_all_family_postprocessing' if all_families else 'auxiliary_cpu_postprocessing'
if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('BLOCKED_RESOURCE: CPU diagnostic RAM admission')
a=json.loads((repo/'reports/eia_development_acquisition.json').read_text());origins=auxiliary_rows(a['records'])
rows=[];contexts=[]
for i in range(25,len(origins)):
    context=origins.iloc[i-25:i+1]
    if (pd.to_datetime(context.decision_time,utc=True).diff().dropna()>pd.Timedelta(days=10)).any():raise ValueError('Calendar gap')
    rows.append(origins.iloc[i].to_dict());contexts.append(context.x.tolist())
frame=pd.DataFrame(rows);frame['sequence_index']=np.arange(len(frame));x=np.array(contexts)
# Match the published canonical identity of supervised endpoints; null endpoints stay in forecasts.
eligible=frame[frame.y.notna()].reset_index(drop=True).copy();eligible['sequence_index']=np.arange(len(eligible))
canonical=canonical_data_id(a,eligible,x[frame.y.notna()])
snapshot=runtime/'artifacts/model_input_snapshots'/canonical;validate_bundle(snapshot)
alias=json.loads((snapshot/'receipt.json').read_text())['metadata']['legacy_data_id']
parents=[]
for directory in (runtime/'artifacts/auxiliary_runs').iterdir():
    if not (directory/'receipt.json').exists():continue
    receipt=validate_bundle(directory);document=json.loads((directory/'results.json').read_text())
    if (document['protocol'].get('compute_scope')==('all' if all_families else 'cpu_only') and document.get('state')=='SUCCEEDED_DIAGNOSTIC'
        and {r['data_id'] for r in document['results'] if r['state']=='SUCCEEDED'}<=set([canonical,alias])):
        parents.append((directory,receipt,document))
if not parents:raise RuntimeError('BLOCKED_DATA: corrected immutable CPU development snapshot required')
parent,receipt,document=sorted(parents,key=lambda item:str(item[0]))[0]
families=document['protocol']['families'] if all_families else [f for f in document['protocol']['families'] if f in CPU_FAMILIES]
recipes={family:next(r for r in document['results'] if r.get('outer_year')==2022 and r.get('family')==family and r.get('seed')==11 and r.get('state')=='SUCCEEDED') for family in families}
cost_results=list(document['results']);capacity_method_id=None
capacity_path=repo/'reports/auxiliary_capacity_results.json'
if all_families and capacity_path.exists():
    capacity=json.loads(capacity_path.read_text())
    if capacity.get('state')=='SUCCEEDED_DIAGNOSTIC' and capacity['protocol']['data_id']==canonical:
        families=list(families)+capacity['protocol']['families'];capacity_method_id=capacity['protocol']['method_id']
        cost_results+=capacity['results']
        for family in capacity['protocol']['families']:recipes[family]=next(r for r in capacity['results'] if r['outer_year']==2022 and r['family']==family and r['seed']==11)
cutoff=pd.Timestamp('2022-12-31T23:59:59.999999999Z')
testing=frame[(frame.decision_time>='2023-01-01')&(frame.decision_time<'2024-01-01')]
intervals=testing.dropna(subset=['label_start','label_end'])
training=purged_training(frame[frame.y.notna()],cutoff,list(zip(intervals.label_start,intervals.label_end)))
arrays=preprocess(x,training,testing,cutoff);tx,vx,scale,normalizer,transform=arrays
seeds=[11,37,71,103,149]
implementation=record_training_dependencies(repo,runtime);current_code=code_hash(repo)
numeric=numeric_dependencies(training_dependencies(repo));operator_id=digest(numeric)
fit_input_id=digest({'tx':tx.tolist(),'y':training.y.tolist(),'scale':scale,'normalizer':normalizer,'transform':transform.manifest(),'cutoff':str(cutoff)})
prediction_input_id=digest({'raw_contexts':x[testing.sequence_index].tolist(),'grid':testing[['decision_time','asset']].to_dict('records')})
prior_path=repo/'reports'/(prefix+'.json')
previous=json.loads(prior_path.read_text()) if prior_path.exists() else {}
cpu_previous=json.loads((repo/'reports/auxiliary_cpu_postprocessing.json').read_text()) if all_families and (repo/'reports/auxiliary_cpu_postprocessing.json').exists() else {}
previous_members={(c['candidate_id'].removeprefix('aux_cpu_').removeprefix('aux_all_'),int(m['member_id'])):m for c in cpu_previous.get('candidates',[])+previous.get('candidates',[]) for m in c['members']}
new_fits=0;reused_fits=0
config={'study_id':'sgqx-v3','scope':'retrospective_Auxiliary_TierB_CPU_selection_calibration_diagnostic','families':families,'seeds':seeds,
        'fit_cutoff':cutoff.isoformat(),'hyperparameters':'2022 earlier inner validation only; no 2023 parameter/seed choice',
        'normalizer_id':normalizer,'canonical_data_id':canonical,'source_cpu_snapshot_receipt':digest(receipt),
        'source_cpu_snapshot':str(parent),'forecasts':len(testing),'n_training_dates':int(training.decision_time.nunique()),
        'proposed_CUDA_family_pending':not all_families,'choose_winner':False,'reserved_access':False,'prospective':False,
        'calibration_source_gaps_retained':True,'unit':'million_barrels'}
config.update(fit_input_id=fit_input_id,prediction_input_id=prediction_input_id,numeric_operator_id=operator_id)
if capacity_method_id:config['capacity_method_id']=capacity_method_id
atomic_json(repo/'reports'/(prefix+'_protocol.json'),config)
candidates=[];calibration={'candidates':{}};comparisons=[];members_by_family={}
minimum=json.loads((repo/'configs/statistical_contract.json').read_text())['calibration']['minimum_distinct_dates']
budget=None
if all_families:
    from signalforge.experiments import ComputeBudget
    budget=ComputeBudget(runtime/'ledger/development_compute.sqlite',document['protocol']['budget_seconds'])
    cost=sum(5*2*max(r.get('seconds',300) for r in cost_results if r.get('family')==f and r.get('outer_year')==2022 and r.get('state')=='SUCCEEDED') for f in families if f not in CPU_FAMILIES)
    bill={'created_at':now(),'state':'READY' if cost<=budget.remaining else 'PAUSED_BUDGET','gpu_fit_recipes':5*len(set(families)-CPU_FAMILIES),
        'expected_gpu_seconds_conservative':cost,'remaining_gpu_seconds':budget.remaining,'budget_ceiling':budget.ceiling,
        'method':'twice largest measured 2022 outer fit per family; same epochs, fixed 2022 inner-only hyperparameters','reserved_access':False}
    atomic_json(repo/'reports'/(prefix+'_bill.json'),bill)
    if bill['state']!='READY':raise RuntimeError('PAUSED_BUDGET: measured finalist-fit bill exceeds persistent remaining allowance')
lease=gpu_lease(runtime) if all_families else nullcontext()
with lease:
 for family in families:
    recipe=recipes[family]['trial'];members=[];predictions={}
    for seed in seeds:
        identity=digest({'stage':'cpu_selection_calibration_fit' if family in CPU_FAMILIES else 'cuda_selection_calibration_fit','data_id':canonical,'numeric_operator_id':operator_id,'fit_input_id':fit_input_id,'family':family,'seed':seed,'trial':recipe,
            **({'capacity_method_id':capacity_method_id} if family.startswith('concat_capacity_') else {})})
        directory=runtime/'artifacts/auxiliary'/identity
        old=previous_members.get((family,seed))
        if old and family in CPU_FAMILIES and not (directory/'receipt.json').exists():
            parent=(runtime/old['relative_bundle']).resolve()
            if not parent.is_relative_to(runtime.resolve()):raise ValueError('CPU cache parent outside runtime')
            parent_receipt=validate_bundle(parent)
            if digest(parent_receipt)!=old['receipt_id']:raise ValueError('CPU cache parent receipt changed')
            metric=json.loads((parent/'metrics.json').read_text());manifest=json.loads((parent/'transform.json').read_text())
            dependency=runtime/'artifacts/model_dependency_manifests'/(parent_receipt['metadata']['code_hash']+'-v2')
            validate_bundle(dependency);record=json.loads((dependency/'dependencies.json').read_text())
            expected={'family':family,'seed':seed,'trial':recipe,'data_id':canonical,'normalizer_id':normalizer,'scale':scale,'fit_cutoff':cutoff.isoformat()}
            if (metric.get('state')=='SUCCEEDED' and all(metric.get(k)==v for k,v in expected.items()) and
                manifest==transform.manifest() and numeric_dependencies(record['dependencies'])==numeric):
                directory=parent;identity=metric['run_id']
        if (directory/'receipt.json').exists():
            member_receipt=validate_bundle(directory);mu,q=infer_bundle(directory,x[testing.sequence_index]);reused_fits+=1
        elif family.startswith('concat_capacity_'):
            from signalforge.capacity_development import fit_capacity_artifact
            actual=testing[testing.y.notna()];actual_arrays=preprocess(x,training,actual,cutoff)
            fit_capacity_artifact(repo,runtime,budget,family,recipe,seed,training,actual,actual_arrays,identity,canonical)
            member_receipt=validate_bundle(directory);mu,q=infer_bundle(directory,x[testing.sequence_index]);new_fits+=1
        elif family not in CPU_FAMILIES:
            from signalforge.development import fit_artifact
            actual=testing[testing.y.notna()]
            actual_arrays=preprocess(x,training,actual,cutoff)
            fit_artifact(repo,runtime,budget,family,recipe,seed,training,actual,actual_arrays,identity,current_code,canonical,raw_sequences=x)
            member_receipt=validate_bundle(directory);mu,q=infer_bundle(directory,x[testing.sequence_index]);new_fits+=1
        else:
            new_fits+=1
            model,mu,q,seconds=fit_predict(family,tx,training.y.to_numpy(),scale,vx,seed,recipe['regularization'],recipe['neural'])
            metric={'state':'SUCCEEDED','family':family,'seed':seed,'trial':recipe,'data_id':canonical,'normalizer_id':normalizer,'scale':scale,'run_id':identity,
                    'seconds':seconds,'evidence_kind':'retrospective_selection_calibration_diagnostic','n_train_dates':len(training),'actual_CUDA_used':False,
                    'fit_cutoff':cutoff.isoformat(),'hyperparameter_parent_run_id':recipes[family]['run_id'],'reserved_access':False}
            saved={'decision_time':pd.to_datetime(testing.decision_time,utc=True).astype(str).tolist(),'asset':testing.asset.tolist(),'mean':mu.tolist(),'quantiles':q.tolist(),'prospective':False}
            with tempfile.TemporaryDirectory(dir=runtime/'tmp') as temporary:
                path=Path(temporary)/'model.bin';model.save(path)
                member_receipt=commit_bundle(directory,{'model.bin':path.read_bytes(),'metrics.json':metric,'predictions.json':saved,'transform.json':transform.manifest()},
                    {'code_hash':current_code,'training_implementation_id':implementation,'data_id':canonical,'run_id':identity,'stage':'retrospective_CPU_selection_calibration'})
        members.append({'member_id':str(seed),'receipt_id':digest(member_receipt),'relative_bundle':str(directory.relative_to(runtime))})
        predictions[str(seed)]=(mu,q)
    candidate={'candidate_id':('aux_all_' if all_families else 'aux_cpu_')+family,'track':'Auxiliary-C','members':members,'weights':[.2]*5}
    ensemble=FrozenEnsemble([str(seed) for seed in seeds],[.2]*5);mu,q=ensemble.predict(predictions);identity=candidate_ensemble_id(candidate)
    candidates.append(candidate);members_by_family[family]=members
    selection=(pd.to_datetime(testing.decision_time,utc=True)<pd.Timestamp('2023-07-01T00:00Z'))
    cal_scope=~selection
    available=pd.to_datetime(testing.label_available_at,utc=True)
    y=testing.y.to_numpy();valid_target=np.isfinite(y)
    masks={'selection':selection.to_numpy()&valid_target&available.le(pd.Timestamp('2023-06-30T23:59:59.999999999Z')).to_numpy(),
           'calibration':cal_scope.to_numpy()&valid_target&available.le(pd.Timestamp('2023-12-31T23:59:59.999999999Z')).to_numpy()}
    fit=masks['calibration']
    calibrator=Calibrator().fit(y[fit],q[fit],testing.decision_time[fit],testing.label_available_at[fit],identity,
                               '2023-07-01T00:00Z','2023-12-31T23:59:59.999999999Z',minimum)
    calibration['candidates'][candidate['candidate_id']]=calibrator.manifest()
    processed=calibrator.transform(q[cal_scope],identity)
    forecast={'candidate_id':candidate['candidate_id'],'ensemble_id':identity,'grid':testing[['decision_time','asset']].to_dict('records'),
              'prediction_input_id':prediction_input_id,'fit_input_id':fit_input_id,
              'raw_mean':mu.tolist(),'raw_quantiles':q.tolist(),'calibration_grid':testing[cal_scope][['decision_time','asset']].to_dict('records'),
              'calibration_before_rearrangement':processed['before_rearrangement'].tolist(),'calibration_processed':processed['processed'].tolist(),
              'calibration_crossing_rows':processed['crossing_rows'],'processed_fit_window_scores_are_not_generalization_evidence':True,'prospective':False}
    forecast_id=digest({'ensemble':identity,'calibrator':calibrator.id,'prediction_input_id':prediction_input_id,'grid':forecast['grid']})
    commit_bundle(runtime/'artifacts'/prefix/forecast_id,{'forecasts.json':forecast,'calibrator.json':calibrator.manifest()},
                  {'ensemble_id':identity,'data_id':canonical,'source_gaps_retained':True,'qualified_for_final':False})
    for name,mask in masks.items():
        scope=selection.to_numpy() if name=='selection' else cal_scope.to_numpy()
        comparisons.append({'family':family,'window':name,'forecast_dates':int(scope.sum()),'mature_qualified_target_dates':int(mask.sum()),
             'missing_or_immature_target_dates':int(scope.sum()-mask.sum()),'raw_normalized_pinball':float(quantile_loss(y[mask],q[mask],QUANTILES,scale).mean()),
             'raw_normalized_mse':float(np.mean(((y[mask]-mu[mask])/scale)**2)),
             'raw_interval_90_coverage':float(np.mean((y[mask]>=q[mask,0])&(y[mask]<=q[mask,-1]))),
             'forecast_artifact_id':forecast_id,'calibration_support':calibrator.support if name=='calibration' else None,
             'scope':'retrospective_TierB_diagnostic_with_explicit_incomplete_target_denominator'})
    atomic_json(repo/'reports'/(prefix+'_progress.json'),{'state':'RUNNING','families_completed':len(candidates),'required':len(families),'final_access':False})
    print(family,'five registered seed fits complete',flush=True)
if budget is not None:budget.close()
result={'state':'SUCCEEDED_ALL_FAMILY_DIAGNOSTIC_INCOMPLETE_CALIBRATION_SOURCE' if all_families else 'SUCCEEDED_CPU_DIAGNOSTIC_INCOMPLETE_CALIBRATION_SOURCE','protocol':config,'comparison_table':comparisons,'candidates':candidates,
        'new_model_fits':new_fits,'verified_reused_model_fits':reused_fits,
        'calibration':calibration,'ensemble':{'candidate_ensemble_ids':{c['candidate_id']:candidate_ensemble_id(c) for c in candidates}},
        'source_failures_and_unknown_targets':testing[testing.y.isna()][['decision_time','target_issue_date','target_status']].replace({np.nan:None}).to_dict('records'),
        'qualified_for_final':False,'no_winner_or_seed_selection':True,'all_five_seeds':True,'final_access':False,'prospective':False,'elapsed_seconds':time.monotonic()-started}
artifact_id=digest({'config':config,'member_receipts':[m['receipt_id'] for c in candidates for m in c['members']],'calibration_ids':[v['calibration_id'] for v in calibration['candidates'].values()]})
result['immutable_artifact_id']=artifact_id
commit_bundle(runtime/'artifacts'/(prefix+'_runs')/artifact_id,{'diagnostic.json':result},{'data_id':canonical,'final_access':False})
atomic_json(repo/'reports'/(prefix+'.json'),result)
atomic_json(repo/'reports'/(prefix+'_progress.json'),{'state':result['state'],'families_completed':len(candidates),'required':len(families),'final_access':False})
print(json.dumps({'state':result['state'],'rows':len(comparisons),'models':len(families)*len(seeds),'elapsed_seconds':result['elapsed_seconds']}))
