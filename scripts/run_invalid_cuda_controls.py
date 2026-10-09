"""Real primary CUDA-tree null/leakage controls; never promotion candidates."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from signalforge.runtime import paths,digest,file_hash,atomic_json,validate_bundle,commit_bundle,now,ensure_gpu_owner
from signalforge.resources import research_gpu_stage
from signalforge.models import CudaTree,QUANTILES
from signalforge.data import auxiliary_rows,eia_events
from signalforge.auxiliary import sequences
from signalforge.development import preprocess
from signalforge.panels import track_folds
from signalforge.diagnostics import rebuild_physical_features
from signalforge.experiment_registry import invalid_control
from signalforge.statistics import calendar_indices,paired_statistics
from signalforge.metrics import quantile_loss

repo,runtime=paths();core=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
if core.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: complete CUDA baseline required')
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text());original=auxiliary_rows(acquisition['records'])
frame,raw=sequences(original)
if pd.to_datetime(frame.decision_time,utc=True).ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved controls denied')
wrong_frame,wrong=sequences(rebuild_physical_features(original,invalid_control(eia_events(acquisition['records']),'wrong_release','development')))
assert wrong_frame.decision_time.tolist()==frame.decision_time.tolist()
sentinel=np.concatenate([raw,np.zeros((*raw.shape[:2],1))],-1);sentinel[:,-1,-1]=frame.y.to_numpy()
folds,blocked=track_folds(frame,json.loads((repo/'configs/study.json').read_text()),'Auxiliary-C');assert not blocked
parents=[r for r in core['results'] if r['family']=='lightgbm'];assert len(parents)==15
controls={'train_target_permutation':raw,'wrong_release_clock':wrong,'future_target_sentinel':sentinel}
protocol={'experiment':'E04','family':'lightgbm','partition':'development','seeds':[11,37,71],
    'controls':list(controls),'recipe':'same reference earlier-inner-selected regularization; 100 CUDA rounds per objective',
    'new_HPO_trials':0,'new_pilot_fits':0,'normalizer':'unchanged mature original training targets',
    'sentinel_dimension_change':'one deliberately invalid extra target feature; not architecture-value evidence',
    'promotion_eligible':False,'reserved_access':False,'data_id':parents[0]['data_id'],
    'latest_revision':'BLOCKED_ORIGINAL_VINTAGE: no authenticated revision sequence',
    'implementation_id':digest({n:file_hash(repo/'src/signalforge'/(n+'.py')) for n in ['models','features','diagnostics','experiment_registry']})}
protocol_id=digest(protocol);atomic_json(repo/'reports/auxiliary_invalid_cuda_controls_protocol.json',protocol)
reservation=2*sum(r['seconds'] for r in parents)*len(controls)+120
atomic_json(repo/'reports/auxiliary_invalid_cuda_controls_bill.json',{'state':'RESOURCE_ADMISSION_PENDING','GPU_seconds_reserved':reservation,
    'maximum_new_control_fits':45,'basis':'twice measured matching reference CUDA fit times plus reload overhead','ceiling_unchanged':True})
checks=[];losses={c:{} for c in ['reference',*controls]}
new_fits=0
with research_gpu_stage(repo,runtime,'invalid_primary_CUDA_tree_controls',reservation) as admission:
    for year,fold in folds.items():
        cut=pd.Timestamp(f'{year}-01-01T00:00Z')-pd.Timedelta(nanoseconds=1)
        predictions={c:[] for c in losses}
        for seed in protocol['seeds']:
            reference=next(r for r in parents if r['outer_year']==year and r['seed']==seed)
            reference_directory=runtime/'artifacts/auxiliary'/reference['run_id'];parent_receipt=validate_bundle(reference_directory)
            saved=json.loads((reference_directory/'predictions.json').read_text())
            assert pd.to_datetime(saved['decision_time'],utc=True).tolist()==pd.to_datetime(fold['test'].decision_time,utc=True).tolist()
            predictions['reference'].append(saved['quantiles'])
            for control,x in controls.items():
                ensure_gpu_owner()
                tx,vx,scale,normalizer,transform=preprocess(x,fold['train'],fold['test'],cut)
                assert scale==reference['scale'] and normalizer==reference['normalizer_id']
                y=fold['train'].y.to_numpy().copy()
                if control=='train_target_permutation':y=y[np.random.default_rng(seed+year).permutation(len(y))]
                tx=tx.reshape(len(tx),-1);vx=vx.reshape(len(vx),-1)
                identity=digest({'protocol_id':protocol_id,'year':year,'seed':seed,'control':control,'reference_receipt_id':digest(parent_receipt),
                    'tx':tx.tolist(),'vx':vx.tolist(),'training_y':y.tolist(),'normalizer_id':normalizer,'transform':transform.manifest()})
                destination=runtime/'artifacts/invalid_CUDA_controls'/identity
                if (destination/'receipt.json').exists():
                    validate_bundle(destination);stored=json.loads((destination/'predictions.json').read_text())
                else:
                    new_fits+=1
                    start=time.perf_counter();model=CudaTree('lightgbm',rounds=100,seed=seed,regularization=reference['trial']['regularization']).fit(tx,y)
                    mean,q=model.predict(vx)
                    stored={'decision_time':saved['decision_time'],'mean':mean.tolist(),'quantiles':q.tolist(),'seconds':time.perf_counter()-start}
                    with tempfile.TemporaryDirectory(dir=runtime/'tmp') as temp:
                        path=Path(temp)/'model.bin';model.save(path)
                        commit_bundle(destination,{'model.bin':path.read_bytes(),'predictions.json':stored,'transform.json':transform.manifest()},
                            {'experiment':'E04','control':control,'promotion_eligible':False,'evidence_kind':'invalid_development_control',
                             'normalizer_id':normalizer,'reference_receipt_id':digest(parent_receipt),'reserved_access':False})
                restored=CudaTree.load(destination/'model.bin');mean,q=restored.predict(vx)
                delta=max(float(np.max(np.abs(mean-np.asarray(stored['mean'])))),float(np.max(np.abs(q-np.asarray(stored['quantiles'])))))
                if delta>1e-10:raise ValueError('CUDA invalid-control reload mismatch')
                if not np.isfinite(q).all() or not np.isfinite(mean).all():raise ValueError('Invalid positive-control forecast')
                predictions[control].append(stored['quantiles']);checks.append({'control':control,'outer_year':year,'seed':seed,'artifact_id':identity,'reload_max_abs_delta':delta})
            print(json.dumps({'outer_year':year,'seed':seed,'control_bundles_completed':len(checks)}),flush=True)
        for control,values in predictions.items():
            score=quantile_loss(fold['test'].y.to_numpy(),np.mean(values,axis=0),QUANTILES,scale)
            losses[control].update(dict(zip(saved['decision_time'],map(float,score))))
dates=sorted(losses['reference']);assert len(dates)==260
indices,metadata=calendar_indices(dates,8,2000);table=[]
for control in controls:
    assert sorted(losses[control])==dates
    left=[losses['reference'][d] for d in dates];right=[losses[control][d] for d in dates]
    table.append({'control':control,'reference_pinball':float(np.mean(left)),'control_pinball':float(np.mean(right)),
        'paired_reference_minus_control':paired_statistics(left,right,indices,metadata),'promotion_eligible':False,'n_market_dates':len(dates)})
result={'state':'SUCCEEDED_NONPROMOTABLE_CUDA_CONTROLS','protocol':protocol,'comparison_table':table,'checks':checks,
    'n_actual_control_bundles':len(checks),'n_market_dates':len(dates),'actual_CUDA_training_used':True,
    'new_fits_this_execution':new_fits,'resource_admission':admission,
    'tree_inference_backend':'LightGBM host prediction; actual designated training uses CUDA','reserved_access':False,'created_at':now()}
identity=digest(result);commit_bundle(runtime/'artifacts/invalid_CUDA_control_runs'/identity,{'results.json':result},{'protocol_id':protocol_id})
atomic_json(repo/'reports/auxiliary_invalid_cuda_controls.json',result);print(json.dumps(table,indent=2))
atomic_json(repo/'reports/auxiliary_invalid_cuda_controls_bill.json',{'state':'COMPLETED_VERIFIED_MODEL_BUNDLES',
    'GPU_seconds_reserved':reservation,'actual_control_bundles':len(checks),'new_fits_this_execution':new_fits,
    'resource_admission':admission,'ceiling_unchanged':True})
