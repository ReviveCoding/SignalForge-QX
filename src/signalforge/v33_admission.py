"""Pure new-study resource admission. This module cannot start a worker."""
import math
from pathlib import PurePosixPath
from .runtime import digest

PROTOCOL='sgqx-v3.3-retrospective-sia-diagnostic'
LEDGER='/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/ledger/v33_sia_diagnostic_compute.sqlite'

def combinations(config):
    rows=[]
    for track in ['Main-A','Nested-B']:
        spec=config['tracks'][track]
        for year in spec['years']:
            for family in ['mlp','lightgbm','rgmf_gru']:
                for seed in [11,37,71]:
                    rows.append({'track':track,'year':year,'information':spec['information'],
                                 'family':family,'seed':seed,'settings':config['families'][family]})
    if len(rows)!=36 or config['planned_fits']!=36 or config['seeds']!=[11,37,71] or config['tracks']!={'Main-A':{'years':[2019,2020],'information':'I3'},'Nested-B':{'years':[2021,2022],'information':'I4'}}:
        raise ValueError('Exact approved 36-combination design required')
    return rows

def checked_plan(plan):
    payload={k:v for k,v in plan.items() if k!='plan_id'}
    if digest(payload)!=plan.get('plan_id') or plan.get('protocol_id')!=PROTOCOL or plan.get('reserved_access') is not False:
        raise PermissionError('Plan identity/scope integrity')
    expected=combinations(plan['registered_config'])
    if [{k:r[k] for k in ['track','year','information','family','seed','settings']} for r in plan['fits']]!=expected:
        raise PermissionError('Frozen grid mismatch')
    return payload

def admission(plan,bill=None,authorization=None,qualification=None):
    checked_plan(plan); reasons=[]
    if any(not r.get('fold_id') or not r.get('source_id') or not r.get('sia_manifest_id') or
           not r.get('normalizer_id') or r.get('feasible') is not True for r in plan['fits']):
        reasons.append('EXACT_FOLD_SOURCE_TRANSFORM_QUALIFICATION')
    if bill is None:
        reasons.extend(['SEPARATE_MEASURED_PILOT_AND_IMMUTABLE_BILL','NUMERIC_NEW_GPU_CEILING'])
    else:
        ceiling=bill.get('ceiling_seconds')
        try: bill_hash=digest({k:v for k,v in bill.items() if k!='bill_id'})
        except (ValueError,TypeError): bill_hash=None
        if bill_hash is None or bill.get('bill_id')!=bill_hash:
            reasons.append('IMMUTABLE_BILL_CONTENT_HASH')
        if bill.get('measurement_source_hash')!=plan.get('source_tree_hash'):
            reasons.append('MEASURED_PILOT_CURRENT_CODE_BINDING')
        numeric=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0
        if not numeric(ceiling):reasons.append('NUMERIC_NEW_GPU_CEILING')
        projected=bill.get('conservative_projected_seconds')
        if not numeric(projected) or not numeric(ceiling) or projected>ceiling:
            reasons.append('CONSERVATIVE_PROJECTED_COST_WITHIN_CEILING')
        if bill.get('plan_id')!=plan['plan_id'] or bill.get('protocol_id')!=PROTOCOL or bill.get('outcome_blind') is not True or bill.get('measured_pilot') is not True or not bill.get('pilot_receipt_sha256'):
            reasons.append('MEASURED_BILL_IDENTITY_AND_PROVENANCE')
        if bill.get('ledger_path')!=LEDGER or '..' in PurePosixPath(bill.get('ledger_path','')).parts or bill.get('parent_ledger_reuse') is not False:
            reasons.append('SEPARATE_V33_EXT4_LEDGER')
        if bill.get('planned_fits')!=36 or not numeric(bill.get('safety_margin')) or bill['safety_margin']<1.25:
            reasons.append('FIT_COUNT_AND_REGISTERED_SAFETY_MARGIN')
    if authorization is None:
        reasons.append('EXPLICIT_USER_NUMERIC_COST_AUTHORIZATION')
    elif bill is None or authorization.get('protocol_id')!=PROTOCOL or authorization.get('plan_id')!=plan['plan_id'] or authorization.get('ceiling_seconds')!=bill.get('ceiling_seconds') or authorization.get('ledger_path')!=LEDGER or authorization.get('user_cost_authorized') is not True:
        reasons.append('EXPLICIT_USER_NUMERIC_COST_AUTHORIZATION')
    if qualification is None or qualification.get('passed') is not True or qualification.get('scope')!='v33_execution' or qualification.get('source_tree_hash')!=plan.get('source_tree_hash') or qualification.get('plan_id')!=plan['plan_id'] or qualification.get('gpu_contract_passed') is not True:
        reasons.append('CURRENT_V33_SOURCE_BOUND_EXECUTION_VALIDATION')
    return {'state':'BLOCKED_RESOURCE_ADMISSION' if reasons else 'ELIGIBLE_FOR_LEASE_CHECK_ONLY',
            'blocked_reasons':reasons,'plan_id':plan['plan_id'],'training_started':False,
            'ledger_path':LEDGER,'numeric_gpu_ceiling':bill.get('ceiling_seconds') if bill else None,
            'old_remaining_seconds_not_authorization':True,'reserved_access':False}

def require_admission(*args,**kwargs):
    result=admission(*args,**kwargs)
    if result['blocked_reasons']:raise PermissionError(';'.join(result['blocked_reasons']))
    return result

def fold_support(training, minimum_dates):
    import pandas as pd
    dates=pd.to_datetime(training.decision_time,utc=True)
    if dates.max()>=pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError('Reserved support count')
    count=int(dates.nunique())
    return {'passed':count>=minimum_dates,'independent_mature_context_dates':count,
            'minimum':minimum_dates,'reason':None if count>=minimum_dates else
            'POST_CONTEXT_TRAINING_DATES_'+str(count)+'_LT_REGISTERED_'+str(minimum_dates)}