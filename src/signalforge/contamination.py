"""Inspection history is append-only; a renamed split is not a fresh cohort."""
import pandas as pd
from .runtime import digest,commit_bundle


def assert_unused_period(history,start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or not start<end:raise ValueError('Aware ordered half-open cohort interval required')
    for record in history:
        left,right=pd.Timestamp(record['start']),pd.Timestamp(record['end'])
        if left.tzinfo is None or right.tzinfo is None or not left<right:raise ValueError('Invalid inspection history')
        if max(start,left)<min(end,right):raise PermissionError('Previously inspected dates cannot become untouched by renaming a split')
    return True


def record_inspection(runtime,record,evidence_bytes=None):
    assert_unused_period([],record['start'],record['end'])
    required={'purpose','evidence_path','evidence_sha256','scientific_partition'}
    if not required<=set(record) or any(not record[k] for k in required):raise ValueError('Actual inspection evidence required')
    from pathlib import Path
    import hashlib
    payload=Path(record['evidence_path']).read_bytes() if evidence_bytes is None else evidence_bytes
    if hashlib.sha256(payload).hexdigest()!=record['evidence_sha256']:raise ValueError('Inspection evidence checksum mismatch')
    identity=digest({'schema':'inspection_record_v2','record':record})
    commit_bundle(runtime/'artifacts/inspection_history'/identity,{'inspection.json':record,'evidence.json':payload},{'inspection_id':identity})
    return identity


def history_records(runtime):
    import json
    from .runtime import validate_bundle
    directory=runtime/'artifacts/inspection_history';records=[]
    for bundle in sorted(directory.iterdir()) if directory.exists() else []:
        if not (bundle/'receipt.json').exists():raise PermissionError('Incomplete inspection history requires reconciliation')
        validate_bundle(bundle);records.append(json.loads((bundle/'inspection.json').read_text()))
    return records
