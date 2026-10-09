"""One immutable scientifically authorized batch, predictions before label reads."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .integrity import authorize_final
from .runtime import atomic_json,commit_bundle,validate_bundle,file_hash,digest,now,gpu_lease
from .inference import infer_candidates
from .models import QUANTILES
from .evaluation import score_grid,holm
from .statistics import calendar_indices,paired_statistics
from .frozen_processing import validate_frozen_postprocessing,apply_frozen_postprocessing


def access_log(runtime,event):
    import fcntl
    path=runtime/'ledger/final_access_events.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(json.dumps({'at':now(),**event},sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def validate_prediction_resume(record,candidate,fallback,grid,calibration):
    """A checksum alone cannot qualify a wrong-grid or wrong-calibrator cache."""
    if record.get('candidate_id')!=candidate['candidate_id'] or record.get('track')!=candidate['track'] or record.get('grid')!=grid:
        raise PermissionError('Frozen prediction resume candidate/track/grid mismatch')
    failure=record.get('failure_type')
    if failure is not None and (not isinstance(failure,str) or not failure):raise PermissionError('Invalid frozen failure state')
    if record.get('fallback_candidate_id')!=fallback['candidate_id']:raise PermissionError('Frozen fallback changed')
    mu=np.asarray(record['mean']);raw=np.asarray(record['raw_quantiles'])
    if mu.shape!=(len(grid),) or raw.shape!=(len(grid),5) or not np.isfinite(mu).all() or not np.isfinite(raw).all():
        raise PermissionError('Frozen prediction resume shape/finite failure')
    processed=apply_frozen_postprocessing(fallback if failure else candidate,mu,raw,calibration)
    if (record.get('ensemble_id')!=processed['ensemble_id'] or record.get('calibration_crossing_rows')!=processed['calibration_crossing_rows']
        or record.get('native_prediction_coverage')!=(0. if failure else 1.)
        or not np.array_equal(np.asarray(record['quantiles']),processed['quantiles'])
        or not np.array_equal(np.asarray(record['before_rearrangement']),processed['before_rearrangement'])):
        raise PermissionError('Frozen prediction resume processing/failure policy mismatch')
    return True


def read_reserved(path,kind,freeze_id,repo,runtime):
    permission=runtime/'ledger/final_access.json'
    if not permission.exists() or json.loads(permission.read_text()).get('freeze_id')!=freeze_id:
        raise PermissionError('Verified scientific batch admission required before reserved read')
    if kind not in {'manifest','features','labels'}:raise ValueError('Registered reserved payload kind required')
    path=Path(path).resolve()
    if not path.is_relative_to(runtime.resolve()):raise PermissionError('Reserved payload outside ext4 runtime')
    access_log(runtime,{'event':'READ_BEGIN','kind':kind,'path':str(path),'freeze_id':freeze_id})
    if path.stat().st_size>128*1024**2:raise RuntimeError('BLOCKED_RESOURCE: reserved JSON payload size ceiling')
    payload=path.read_bytes()
    if len(payload)>128*1024**2:raise RuntimeError('BLOCKED_RESOURCE: reserved JSON payload size ceiling')
    import hashlib
    checksum=hashlib.sha256(payload).hexdigest();hash_path=runtime/'ledger'/('reserved_'+kind+'_hash.json')
    if hash_path.exists():
        previous=json.loads(hash_path.read_text())
        if previous!={'freeze_id':freeze_id,'sha256':checksum}:raise PermissionError('Reserved snapshot changed during one-batch resume')
    else:atomic_json(hash_path,{'freeze_id':freeze_id,'sha256':checksum})
    if kind=='labels':
        from .contamination import record_inspection
        record_inspection(runtime,{'start':'2024-01-01T00:00Z','end':'2026-07-01T00:00Z',
            'purpose':'authorized frozen reserved-label read','evidence_path':str(path),'evidence_sha256':checksum,
            'scientific_partition':'authorized_reserved_batch','freeze_id':freeze_id},evidence_bytes=payload)
    result=json.loads(payload)
    access_log(runtime,{'event':'READ_COMMITTED','kind':kind,'sha256':checksum,'freeze_id':freeze_id})
    return result


def execute_frozen(repo,runtime,receipt_path):
    receipt=json.loads(Path(receipt_path).read_text(encoding='utf-8-sig'))
    auth=json.loads((repo/'.local/scientific_authorization.json').read_text(encoding='utf-8-sig'))
    from .integrity import verify_freeze
    freeze_id=verify_freeze(receipt,repo,runtime,auth)
    models=json.loads(Path(receipt['components']['models']['path']).read_text())
    candidates=models['candidates']
    if {c['candidate_id'] for c in candidates}!=set(receipt['registered_candidate_ids']):
        raise PermissionError('Full frozen candidate batch mismatch')
    contrasts=json.loads(Path(receipt['components']['contrast_family']['path']).read_text())['contrasts']
    if {c['id'] for c in contrasts}!=set(receipt['registered_contrast_ids']):raise PermissionError('Full contrast family mismatch')
    ensembles=json.loads(Path(receipt['components']['ensemble']['path']).read_text())
    calibration=json.loads(Path(receipt['components']['calibration']['path']).read_text())
    minimum_dates=json.loads((repo/'configs/statistical_contract.json').read_text())['calibration']['minimum_distinct_dates']
    validate_frozen_postprocessing(candidates,ensembles,calibration,minimum_dates)
    plan=json.loads(Path(receipt['components']['features']['path']).read_text())
    freeze_id=authorize_final(receipt,repo,runtime,auth,'locked')
    completed=runtime/'artifacts/final'/freeze_id
    if (completed/'receipt.json').exists():
        bundle=validate_bundle(completed)
        report=json.loads((completed/'evaluation.json').read_text())
        if (bundle['metadata']!={'freeze_id':freeze_id} or report.get('freeze_id')!=freeze_id or
            report.get('status')!='FROZEN_BATCH_EXECUTED' or
            set(report.get('scores',{}))!=set(receipt['registered_candidate_ids']) or
            {c.get('contrast_id') for c in report.get('comparisons',[])}!=set(receipt['registered_contrast_ids']) or
            report.get('no_model_seed_threshold_selection') is not True):
            raise PermissionError('Completed frozen batch receipt/result identity mismatch')
        # Preserve the original actual_at and immutable result; no second target read.
        atomic_json(repo/'reports/final_evaluation.json',report)
        access_log(runtime,{'event':'COMPLETED_BATCH_REPLAY','freeze_id':freeze_id,'reserved_data_reread':False})
        return report
    # No development command reads this payload. Manifest lookup occurs after scientific admission.
    inputs=read_reserved(plan['reserved_inputs_manifest'],'manifest',freeze_id,repo,runtime)
    features=read_reserved(inputs['features_path'],'features',freeze_id,repo,runtime)
    predictions={};failures=[]
    with gpu_lease(runtime):
        for track,panel in features['tracks'].items():
            rows=panel['rows'];grid=[{'decision_time':r['decision_time'],'asset':r['asset']} for r in rows]
            if digest(grid)!=plan['grid_ids'][track]:raise PermissionError('Outcome-blind frozen grid differs')
            for row in rows:
                decision=pd.Timestamp(row['decision_time']);known=pd.Timestamp(row['max_dependency_available_at'])
                if decision.tzinfo is None or known.tzinfo is None or known>decision or 'y' in row or 'outcome' in row:
                    raise PermissionError('Future or target field in frozen feature inputs')
                if not pd.Timestamp('2024-01-01T00:00Z')<=decision<pd.Timestamp('2026-07-01T00:00Z'):
                    raise PermissionError('Decision outside registered reserved cohort')
            x=np.asarray([r['x'] for r in rows],dtype=float)
            inference_grid={'assets':[r['asset'] for r in rows],
                            'context_eligible':[r['context_eligible'] for r in rows] if all('context_eligible' in r for r in rows) else None}
            if all('context_decision_times' in r for r in rows):
                for row in rows:
                    times=row['context_decision_times']
                    if not times or pd.Timestamp(times[-1])!=pd.Timestamp(row['decision_time']):
                        raise PermissionError('Frozen context must end at its decision origin')
                inference_grid['context_decision_times']=[r['context_decision_times'] for r in rows]
            eligible=[c for c in candidates if c['track']==track]
            fallback_id=models['fallback_candidate_ids'][track]
            fallback=next(c for c in eligible if c['candidate_id']==fallback_id)
            fallback_prediction=infer_candidates([fallback],x,runtime,**inference_grid)[fallback_id]
            for candidate in eligible:
                identity=digest({'freeze':freeze_id,'candidate':candidate['candidate_id'],'track':track})
                directory=runtime/'artifacts/final_predictions'/identity
                if (directory/'receipt.json').exists():
                    cached_receipt=validate_bundle(directory)
                    if cached_receipt['metadata']!={'freeze_id':freeze_id,'candidate_id':candidate['candidate_id']}:raise PermissionError('Frozen prediction receipt metadata mismatch')
                    record=json.loads((directory/'predictions.json').read_text())
                    validate_prediction_resume(record,candidate,fallback,grid,calibration)
                else:
                    try:
                        mu,q=infer_candidates([candidate],x,runtime,**inference_grid)[candidate['candidate_id']]
                        if mu.shape!=(len(rows),) or q.shape!=(len(rows),5) or not np.isfinite(mu).all() or not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():raise ValueError('Invalid frozen forecasts')
                        failure=None
                    except Exception as exc:
                        mu,q=fallback_prediction;failure=type(exc).__name__
                    processed=apply_frozen_postprocessing(fallback if failure else candidate,mu,q,calibration)
                    record={'candidate_id':candidate['candidate_id'],'track':track,'grid':grid,'mean':mu.tolist(),'quantiles':processed['quantiles'].tolist(),
                            'raw_quantiles':q.tolist(),'before_rearrangement':processed['before_rearrangement'].tolist(),
                            'calibration_crossing_rows':processed['calibration_crossing_rows'],'ensemble_id':processed['ensemble_id'],
                            'native_prediction_coverage':0. if failure else 1.,'failure_type':failure,'fallback_candidate_id':fallback_id}
                    commit_bundle(directory,{'predictions.json':record},{'freeze_id':freeze_id,'candidate_id':candidate['candidate_id']})
                    access_log(runtime,{'event':'PREDICTIONS_COMMITTED','candidate_id':candidate['candidate_id'],'freeze_id':freeze_id})
                predictions[candidate['candidate_id']]=record
                if record['failure_type']:failures.append({'candidate_id':candidate['candidate_id'],'error_type':record['failure_type']})
    if set(predictions)!=set(receipt['registered_candidate_ids']):raise PermissionError('Frozen track/candidate predictions incomplete; labels remain unread')
    labels=read_reserved(inputs['labels_path'],'labels',freeze_id,repo,runtime)
    metrics=json.loads(Path(receipt['components']['metrics_slices']['path']).read_text())
    scores={}
    for candidate_id,prediction in predictions.items():
        track=prediction['track'];grid=pd.DataFrame(prediction['grid']);target=pd.DataFrame(labels['tracks'][track]['rows'])
        if target.duplicated(['decision_time','asset']).any():raise ValueError('Duplicate reserved targets')
        target_keys=target[['decision_time','asset']].sort_values(['decision_time','asset']).reset_index(drop=True)
        grid_keys=grid.sort_values(['decision_time','asset']).reset_index(drop=True)
        if not target_keys.equals(grid_keys):raise ValueError('Reserved labels do not exactly match the frozen outcome-blind grid')
        for row in target.to_dict('records'):
            publication=pd.Timestamp(row['label_available_at']);end=pd.Timestamp(row['label_end'])
            if publication.tzinfo is None or end.tzinfo is None or publication<end or publication>pd.Timestamp(now()):
                raise PermissionError('Reserved target not actually mature/public')
        grid=grid.merge(target,on=['decision_time','asset'],validate='one_to_one',how='left')
        if not np.isfinite(grid.y).all():raise ValueError('Frozen grid missing outcomes; no favorable-row deletion')
        scales=np.array([metrics['scales'][track][asset] for asset in grid.asset])
        pred=grid[['decision_time','asset']].copy()
        for j,q in enumerate(QUANTILES):pred[f'q{q:g}']=np.array(prediction['quantiles'])[:,j]
        score=score_grid(grid,pred,pred,QUANTILES,scales)
        raw_pred=pred.copy()
        for j,q in enumerate(QUANTILES):raw_pred[f'q{q:g}']=np.array(prediction['raw_quantiles'])[:,j]
        score['raw_adapter_score']=score_grid(grid,raw_pred,raw_pred,QUANTILES,scales)
        mean_errors=((grid.y.to_numpy()-np.array(prediction['mean']))/scales)**2
        score['normalized_mean_squared_error']=float(pd.DataFrame({'date':grid.decision_time,'error':mean_errors}).groupby('date').error.mean().mean())
        score['calibration_crossing_rows']=prediction['calibration_crossing_rows']
        score['native_prediction_coverage']=prediction['native_prediction_coverage']
        score['fallback_count']=len(grid) if prediction['failure_type'] else 0
        scores[candidate_id]=score
    draws={};comparisons=[];pvalues={}
    for contrast in contrasts:
        left=scores[contrast['left']]['date_losses'];right=scores[contrast['right']]['date_losses']
        if list(left)!=list(right):raise ValueError('Frozen contrast uses unmatched date grid')
        key=digest(list(left))
        if key not in draws:draws[key]=calendar_indices(list(left),8,2000)
        indices,metadata=draws[key]
        result=paired_statistics(list(left.values()),list(right.values()),indices,metadata)
        comparisons.append({'contrast_id':contrast['id'],**result});pvalues[contrast['id']]=result['two_sided_centered_block_p']
    confirmatory={c['id']:pvalues[c['id']] for c in contrasts if c.get('confirmatory') is True}
    declared=json.loads((repo/'configs/comparisons.json').read_text())['confirmatory_families']
    if set(confirmatory)!={c['id'] for c in declared}:raise PermissionError('Complete predeclared Holm family required')
    report={'freeze_id':freeze_id,'status':'FROZEN_BATCH_EXECUTED','scores':scores,'comparisons':comparisons,'Holm':holm(confirmatory),
            'failures':failures,'economics_state':receipt['economics_state'],'no_model_seed_threshold_selection':True,'actual_at':now()}
    commit_bundle(runtime/'artifacts/final'/freeze_id,{'evaluation.json':report},{'freeze_id':freeze_id})
    access_log(runtime,{'event':'OUTPUT_EXPOSED','freeze_id':freeze_id,'candidate_ids':receipt['registered_candidate_ids']})
    atomic_json(repo/'reports/final_evaluation.json',report)
    return report
