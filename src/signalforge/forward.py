"""Real-clock forecast production after an operational freeze, never backdated."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .integrity import forward_clock
from .runtime import now,digest,commit_bundle,validate_bundle,gpu_lease
from .inference import infer_candidates


def validate_forward(features,frozen_at,actual_now):
    decision=features['decision_time'];outcome=features['expected_outcome_available_at']
    forward_clock(frozen_at,decision,outcome,actual_now)
    if pd.Timestamp(features['max_dependency_available_at'])>pd.Timestamp(decision):raise PermissionError('Future forward feature dependency')
    if 'y' in features or 'outcome' in features:raise PermissionError('Forward input cannot contain outcomes')
    if pd.Timestamp(actual_now)-pd.Timestamp(decision)>pd.Timedelta(hours=12):
        raise PermissionError('Missed operational issuance window; no late historical replay')
    return True


def append_mature_outcome(runtime,forecast_directory,outcome,actual_now=None):
    """Commit a separate immutable observation; never overwrite the issued forecast."""
    directory=Path(forecast_directory).resolve();runtime=Path(runtime).resolve()
    if not directory.is_relative_to(runtime/'artifacts/forward'):raise PermissionError('Forward cohort path required')
    receipt=validate_bundle(directory)
    record=json.loads((directory/'forecast.json').read_text())
    if receipt['metadata']!={'freeze_id':record['freeze_id'],'prospective':True}:raise PermissionError('Prospective receipt required')
    current=pd.Timestamp(actual_now or now());issued=pd.Timestamp(record['issued_at'])
    published=pd.Timestamp(outcome['available_at']);end=pd.Timestamp(outcome['label_end'])
    if any(t.tzinfo is None for t in [current,issued,published,end]):raise ValueError('Aware outcome clocks required')
    if not issued<end<=published<=current:raise PermissionError('Outcome was not issued first or is not mature/public')
    if outcome.get('decision_time')!=record['decision_time'] or outcome.get('assets')!=record['assets']:
        raise ValueError('Immutable forecast/outcome grid mismatch')
    values=np.asarray(outcome['values'],dtype=float)
    if values.shape!=(len(record['assets']),) or not np.isfinite(values).all():raise ValueError('Finite full-grid outcomes required')
    import re
    if not outcome.get('raw_hashes') or any(not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h) for h in outcome['raw_hashes']):
        raise ValueError('Actual outcome raw lineage required')
    target=runtime/'artifacts/forward_outcomes'/directory.name
    content={'forecast_receipt_id':digest(receipt),'freeze_id':record['freeze_id'],'outcome':outcome}
    if (target/'receipt.json').exists():
        validate_bundle(target)
        if json.loads((target/'outcome.json').read_text())!=content:raise PermissionError('Append-only outcome already committed; revised version needs separate registered cohort')
        return content
    commit_bundle(target,{'outcome.json':content},{'freeze_id':record['freeze_id'],'forecast_receipt_id':digest(receipt)})
    return content


def produce_once(repo,runtime,asof):
    path=repo/'.local/operational_freeze.json'
    if not path.exists():raise RuntimeError('BLOCKED_DATA: qualified operational freeze absent')
    freeze=json.loads(path.read_text(encoding='utf-8-sig'))
    if freeze.get('status')!='READY_FOR_FORWARD' or freeze.get('qualified_real_development') is not True:
        raise PermissionError('Operational scientific qualification absent')
    from .integrity import verify_qualification
    from .runtime import file_hash
    for name,artifact in freeze['components'].items():
        artifact_path=Path(artifact['path']).resolve()
        if not any(artifact_path.is_relative_to(root.resolve()) for root in [repo,runtime]) or file_hash(artifact_path)!=artifact['sha256']:
            raise PermissionError('Operational frozen component changed: '+name)
    verify_qualification(freeze,repo)
    from .runtime import code_hash
    if freeze.get('source_code_hash')!=code_hash(repo):raise PermissionError('Code changed since operational freeze')
    input_path=(runtime/freeze['current_feature_snapshot']).resolve()
    if not input_path.is_relative_to(runtime.resolve()):raise PermissionError('Forward snapshot outside runtime')
    features=json.loads(input_path.read_text());actual=now()
    if asof!='NOW' and pd.Timestamp(asof)!=pd.Timestamp(features['decision_time']):raise PermissionError('As-of must match the actual registered snapshot')
    validate_forward(features,freeze['frozen_at'],actual)
    freeze_id=digest(freeze);identity=digest({'freeze':freeze_id,'decision':features['decision_time'],'asset_grid':features['assets']})
    directory=runtime/'artifacts/forward'/identity
    if (directory/'receipt.json').exists():validate_bundle(directory);return json.loads((directory/'forecast.json').read_text())
    from .frozen_processing import validate_frozen_postprocessing,apply_frozen_postprocessing
    ensembles=json.loads(Path(freeze['components']['ensemble']['path']).read_text())
    calibration=json.loads(Path(freeze['components']['calibration']['path']).read_text())
    minimum_dates=json.loads((repo/'configs/statistical_contract.json').read_text())['calibration']['minimum_distinct_dates']
    validate_frozen_postprocessing(freeze['candidates'],ensembles,calibration,minimum_dates)
    inference_grid={'assets':features['assets'],'context_eligible':features.get('context_eligible')}
    if 'context_decision_times' in features:
        if any(not times or pd.Timestamp(times[-1])!=pd.Timestamp(features['decision_time']) for times in features['context_decision_times']):
            raise PermissionError('Forward context must end at its actual origin')
        inference_grid['context_decision_times']=features['context_decision_times']
    with gpu_lease(runtime):predictions=infer_candidates(freeze['candidates'],np.asarray(features['x']),runtime,**inference_grid)
    processed={c['candidate_id']:apply_frozen_postprocessing(c,*predictions[c['candidate_id']],calibration) for c in freeze['candidates']}
    emitted=now();validate_forward(features,freeze['frozen_at'],emitted)
    record={'state':'FORECAST_ISSUED_OUTCOME_PENDING','decision_time':features['decision_time'],'issued_at':emitted,
            'expected_outcome_available_at':features['expected_outcome_available_at'],'freeze_id':freeze_id,
            'max_dependency_available_at':features['max_dependency_available_at'],'feature_snapshot_id':digest(features),'assets':features['assets'],
            'predictions':{k:{'mean':v[0].tolist(),'raw_quantiles':v[1].tolist(),'quantiles':processed[k]['quantiles'].tolist(),
                'calibration_crossing_rows':processed[k]['calibration_crossing_rows'],'ensemble_id':processed[k]['ensemble_id']} for k,v in predictions.items()},'observed_outcome':None,'real_orders':False}
    commit_bundle(directory,{'forecast.json':record},{'freeze_id':freeze_id,'prospective':True})
    return record
