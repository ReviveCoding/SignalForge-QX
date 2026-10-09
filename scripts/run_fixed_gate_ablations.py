"""Paired retraining at reference inner-selected recipes, never new HPO candidates."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from signalforge.runtime import paths,digest,file_hash,atomic_json,validate_bundle,commit_bundle,now,ensure_gpu_owner
from signalforge.resources import research_gpu_stage
from signalforge.fixed_gate import FixedGateCUDA
from signalforge.track_neural import raw_source_validity
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences
from signalforge.development import preprocess
from signalforge.panels import track_folds
from signalforge.models import QUANTILES
from signalforge.metrics import quantile_loss
from signalforge.statistics import calendar_indices,paired_statistics

repo,runtime=paths();core=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
if core.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: complete core required')
families=['rgmf_linear','rgmf_gru','rgmf_transformer']
parents=[r for r in core['results'] if r['family'] in families]
assert len(parents)==45 and all(r['state']=='SUCCEEDED' for r in parents)
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
frame,raw=sequences(auxiliary_rows(acquisition['records']));study=json.loads((repo/'configs/study.json').read_text())
folds,blocked=track_folds(frame,study,'Auxiliary-C');assert not blocked
protocol={'experiment':'E03','intervention_type':'retrained_fixed_gate','retrained':True,'families':families,
    'outer_years':sorted(folds),'seeds':[11,37,71],'epochs':100,'gate':'uniform among currently valid experts; exact base-only when all missing',
    'recipe':'same width/LR as reference earlier-inner-selected recipe, same seed, optimizer and full input grid',
    'HPO_new_trials':0,'new_pilot_fits':0,'selection_policy':'isolated paired component test; not optimized fixed-gate architecture selection',
    'parameter_policy':'same experts and gate tensors; gate parameters frozen to zero',
    'promotion_eligible':False,'reserved_access':False,'data_id':core['results'][0]['data_id'],
    'implementation_id':digest({n:file_hash(repo/'src/signalforge'/(n+'.py')) for n in ['fixed_gate','track_neural','models','neural']})}
protocol_id=digest(protocol);atomic_json(repo/'reports/fixed_gate_retrained_protocol.json',protocol)
recipes=[]
for row in parents:
    identity=digest({'protocol_id':protocol_id,'reference_run_id':row['run_id'],'family':row['family'],'year':row['outer_year'],'seed':row['seed'],'trial':row['trial']})
    directory=runtime/'artifacts/fixed_gate_retrained'/identity
    recipes.append((row,identity,directory))
missing=[r for r,_,d in recipes if not (d/'receipt.json').exists()]
reservation=max(60.,2*sum(r['seconds'] for r in missing)+60)
atomic_json(repo/'reports/fixed_gate_retrained_bill.json',{'state':'RESOURCE_ADMISSION_PENDING','GPU_seconds_reserved':reservation,
    'missing_fits':len(missing),'existing_fits':45-len(missing),'basis':'twice measured matching reference fit time plus 60 seconds reload overhead','budget_ceiling_unchanged':True})
results=[];losses={f:{'reference':{},'fixed_gate':{}} for f in families}
with research_gpu_stage(repo,runtime,'retrained_fixed_gate',reservation) as admission:
    for row,identity,directory in recipes:
        ensure_gpu_owner()
        fold=folds[row['outer_year']];cut=pd.Timestamp(f"{row['outer_year']}-01-01T00:00Z")-pd.Timedelta(nanoseconds=1)
        tx,vx,scale,normalizer,transform=preprocess(raw,fold['train'],fold['test'],cut)
        assert normalizer==row['normalizer_id'] and scale==row['scale']
        observed=np.concatenate([np.isfinite(raw[fold['train'].sequence_index])]*2,-1)
        test_observed=np.concatenate([np.isfinite(raw[fold['test'].sequence_index])]*2,-1)
        valid=raw_source_validity(raw[fold['train'].sequence_index],[list(range(2,10))])
        test_valid=raw_source_validity(raw[fold['test'].sequence_index],[list(range(2,10))])
        reference_directory=runtime/'artifacts/auxiliary'/row['run_id'];reference_receipt=validate_bundle(reference_directory)
        reference=json.loads((reference_directory/'predictions.json').read_text())
        if (directory/'receipt.json').exists():
            validate_bundle(directory);saved=json.loads((directory/'predictions.json').read_text());metric=json.loads((directory/'metrics.json').read_text())
        else:
            width,lr=row['trial']['neural'];f=tx.shape[-1]//2;base=[0,1,12,13,f,f+1,f+12,f+13]
            model=FixedGateCUDA(kind=row['family'],width=width,lr=lr,epochs=100,seed=row['seed'],
                base_columns=base,source_columns=[[i for i in range(tx.shape[-1]) if i not in base]],meta_columns=[10,11,f+10,f+11])
            start=time.perf_counter();model.fit(tx,fold['train'].y.to_numpy(),scale,observed=observed,source_valid=valid,
                checkpoint_path=runtime/'checkpoints/fixed_gate_retrained'/(identity+'.pt'))
            mean,q=model.predict(vx,observed=test_observed,source_valid=test_valid)
            if not np.isfinite(mean).all() or not np.isfinite(q).all() or (np.diff(q)<0).any():raise ValueError('Invalid fixed gate prediction')
            saved={'decision_time':[d.isoformat() for d in pd.to_datetime(fold['test'].decision_time,utc=True)],'y':fold['test'].y.tolist(),'mean':mean.tolist(),'quantiles':q.tolist()}
            metric={'family':row['family'],'outer_year':row['outer_year'],'seed':row['seed'],'trial':row['trial'],
                'state':'SUCCEEDED','seconds':time.perf_counter()-start,'normalizer_id':normalizer,'scale':scale,'run_id':identity,
                'reference_run_id':row['run_id'],'parameter_count':model.parameter_count,
                'trainable_parameter_count':sum(p.numel() for p in model.model.parameters() if p.requires_grad),
                'actual_CUDA_used':True,'data_id':protocol['data_id']}
            with tempfile.TemporaryDirectory(dir=runtime/'tmp') as tmp:
                path=Path(tmp)/'model.bin';model.save(path)
                commit_bundle(directory,{'model.bin':path.read_bytes(),'transform.json':transform.manifest(),'metrics.json':metric,'predictions.json':saved},
                    {'protocol_id':protocol_id,'reference_receipt_id':digest(reference_receipt)})
        if saved['decision_time']!=reference['decision_time'] or saved['y']!=reference['y']:raise ValueError('Fixed-gate comparison grid/target mismatch')
        loaded=FixedGateCUDA.load(directory/'model.bin');mean,q=loaded.predict(vx,observed=test_observed,source_valid=test_valid)
        delta=max(float(np.max(np.abs(mean-np.asarray(saved['mean'])))),float(np.max(np.abs(q-np.asarray(saved['quantiles'])))))
        if delta>2e-6:raise ValueError('Fixed gate actual reload mismatch')
        results.append({**metric,'reload_max_abs_delta':delta})
    for family in families:
        for year in sorted(folds):
            group=[r for r in results if r['family']==family and r['outer_year']==year]
            fixed=[];references=[]
            for row in group:
                saved=json.loads((runtime/'artifacts/fixed_gate_retrained'/row['run_id']/'predictions.json').read_text())
                ref=json.loads((runtime/'artifacts/auxiliary'/row['reference_run_id']/'predictions.json').read_text())
                fixed.append(saved['quantiles']);references.append(ref['quantiles'])
            for name,values in [('reference',references),('fixed_gate',fixed)]:
                loss=quantile_loss(np.asarray(saved['y']),np.mean(values,axis=0),QUANTILES,row['scale'])
                losses[family][name].update(dict(zip(saved['decision_time'],map(float,loss))))
table=[]
for family,values in losses.items():
    dates=sorted(values['reference']);assert dates==sorted(values['fixed_gate']) and len(dates)==260
    indices,metadata=calendar_indices(dates,8,2000)
    left=[values['reference'][d] for d in dates];right=[values['fixed_gate'][d] for d in dates]
    table.append({'family':family,'reference_pinball':float(np.mean(left)),'fixed_gate_pinball':float(np.mean(right)),
        'reference_minus_fixed_gate':paired_statistics(left,right,indices,metadata),'n_unique_dates':len(dates)})
result={'state':'SUCCEEDED_RETRAINED_FIXED_GATE_DIAGNOSTIC','protocol':protocol,'results':results,'comparison_table':table,
    'date_losses':losses,'resource_admission':admission,'qualified_for_final':False,'reserved_access':False,'created_at':now()}
identity=digest(result);commit_bundle(runtime/'artifacts/fixed_gate_retrained_runs'/identity,{'results.json':result},{'protocol_id':protocol_id})
atomic_json(repo/'reports/fixed_gate_retrained.json',result);print(json.dumps(table,indent=2))
atomic_json(repo/'reports/fixed_gate_retrained_bill.json',{'state':'COMPLETED_VERIFIED_MODEL_BUNDLES',
    'GPU_seconds_reserved':reservation,'actual_bundles':len(results),'new_fits_this_execution':len(missing),
    'resource_admission':admission,'budget_ceiling_unchanged':True})
