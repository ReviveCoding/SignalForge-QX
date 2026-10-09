"""Dedicated six-fit v2 admission; never uses the 36-fit checker or starts work."""
import math
from .runtime import digest
PROTOCOL='sgqx-v3.3-minimal-sia-execution-v2'
BASE='/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/ledger/'
CAPS={'pilot':300,'sia':1800,'bar':1800}
LEDGERS={k:BASE+'v33_minimal_v2_'+k+'_compute.sqlite' for k in CAPS}
GRID=[(t,y,i,'rgmf_gru',s) for t,y,i in [('Main-A',2020,'I3'),('Nested-B',2022,'I4')] for s in [11,37,71]]

def checked(plan):
    if plan.get('plan_id')!=digest({k:v for k,v in plan.items() if k!='plan_id'}):raise PermissionError('PLAN_HASH')
    if plan.get('protocol_id')!=PROTOCOL or plan.get('reserved_access') is not False:raise PermissionError('SCOPE')
    rows=plan.get('fits',[])
    if [(r['track'],r['year'],r['information'],r['family'],r['seed']) for r in rows]!=GRID:raise PermissionError('SIX_FIT_GRID')
    if plan.get('caps')!=CAPS or plan.get('ledgers')!=LEDGERS or plan.get('parent_ledger_reuse') is not False:raise PermissionError('CAP_LEDGER')
    for r in rows:
        if r['settings']!={'width':16,'lr':.001,'epochs':100} or any(not r.get(k) for k in ['fold_id','source_id','normalizer_id','sia_manifest_id']):raise PermissionError('RECIPE')
    return plan

def authorize(plan,auth,source_hash):
    checked(plan)
    if plan['execution_source_hash']!=source_hash:raise PermissionError('CURRENT_SOURCE_HASH')
    if auth.get('authorization_id')!=digest({k:v for k,v in auth.items() if k!='authorization_id'}):raise PermissionError('AUTH_HASH')
    for k,v in {'protocol_id':PROTOCOL,'plan_id':plan['plan_id'],'caps':CAPS,'ledgers':LEDGERS,'user_execution_waiver':True,'reserved_access':False}.items():
        if auth.get(k)!=v:raise PermissionError('AUTH_BINDING')
    return True

def admit(plan,auth,qual,bill,source_hash):
    authorize(plan,auth,source_hash)
    reasons=[]
    if qual.get('source_tree_hash')!=source_hash or qual.get('plan_id')!=plan['plan_id'] or qual.get('passed') is not True or qual.get('cuda_sia_fit_predict_reload') is not True:reasons.append('CUDA_QUALIFICATION')
    if bill.get('bill_id')!=digest({k:v for k,v in bill.items() if k!='bill_id'}):reasons.append('BILL_HASH')
    value=bill.get('conservative_projected_seconds')
    numeric=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0
    if not numeric(value) or value>1800:reasons.append('PROJECTED_BUDGET')
    if bill.get('source_tree_hash')!=source_hash or bill.get('plan_id')!=plan['plan_id'] or bill.get('ledger_path')!=LEDGERS['sia'] or bill.get('fit_count')!=6 or bill.get('outcome_blind') is not True or bill.get('safety_margin')!=1.25 or not bill.get('pilot_receipt_sha256'):reasons.append('MEASURED_BILL_BINDING')
    if not bill.get('measured_seconds_by_fold') or set(bill['measured_seconds_by_fold'])!={'Main-A:2020','Nested-B:2022'}:reasons.append('BOTH_FOLD_PILOT')
    return {'state':'ADMITTED_SIX_FIT_EXECUTION' if not reasons else 'BLOCKED','blocked_reasons':reasons,'reserved_access':False}
