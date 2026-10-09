"""Small stdlib handoff-contract helpers, not the complete research engine."""
from __future__ import annotations
import hashlib,json,math
from pathlib import PurePosixPath

def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def stage_key(stage,upstream,configuration,code_hash,environment_id=None):
    if stage not in {'raw','canonical','features','fit','prediction','postprocess','evaluation','report'}:
        raise ValueError('Unknown stage')
    # Caller supplies only stage-relevant config. Downstream config must not enter a fit key.
    if stage=='fit' and any(k in configuration for k in ('report_template','calibration_hash','postprocessing_hash')):
        raise ValueError('Downstream configuration contaminates fit identity')
    return stable_hash({'stage':stage,'upstream':upstream,'configuration':configuration,'code':code_hash,'environment':environment_id})

def validate_archive_member(name):
    if not name or '\\' in name or ':' in name or name.startswith('/'):
        raise ValueError('Unsafe archive path')
    parts=PurePosixPath(name).parts
    if '..' in parts or parts[0]!='SignalForge-QX':raise ValueError('Unexpected archive root or traversal')
    return parts

def topological_order(phases):
    ids=[p['id'] for p in phases]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate phase')
    done=[];remaining={p['id']:set(p['depends_on']) for p in phases}
    if any(not ds<=set(ids) for ds in remaining.values()):raise ValueError('Missing dependency')
    while remaining:
        ready=sorted(x for x,ds in remaining.items() if ds<=set(done))
        if not ready:raise ValueError('Cycle')
        done.extend(ready)
        for x in ready:del remaining[x]
    return done

def convex_fusion(experts,gates):
    if not experts or len(experts)!=len(gates):raise ValueError('Dimension mismatch')
    if any(not math.isfinite(g) or g<0 for g in gates) or not math.isclose(sum(gates),1,abs_tol=1e-10):raise ValueError('Invalid simplex')
    n=len(experts[0])
    if n==0:raise ValueError("Empty quantile vector")
    if any(len(q)!=n or any(not math.isfinite(x) for x in q) or any(a>b for a,b in zip(q,q[1:])) for q in experts):raise ValueError('Invalid expert quantiles')
    return [sum(g*q[j] for q,g in zip(experts,gates)) for j in range(n)]

def final_allowed(auth,receipt):
    return bool(auth.get('authorized') is True and auth.get('study_id')=='sgqx-v3'
      and auth.get('scope')=='one_registered_frozen_batch_after_all_track_gates'
      and receipt.get('status')=='READY_FOR_FINAL'
      and receipt.get('study_id')=='sgqx-v3'
      and receipt.get('all_required_predictive_gates_passed') is True
      and receipt.get('protocol_hash'))

def check_common_comparison(left,right):
    for key in ['track','price_mode','tier','decision_asset_grid','horizon','normalizer_id']:
        if key not in left or left.get(key)!=right.get(key):raise ValueError(f'Unmatched {key}')
    return True
