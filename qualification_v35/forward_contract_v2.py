"""Prediction readiness independent of P1; pure prepared-bundle validation only."""
from qualification_v35.core import clock,validate_forward
PREDICTION_REQUIREMENTS=['candidate_hashes_verified','benchmark_hashes_verified','normalizers_verified','all_source_versions_tier_a','independent_reviewer','thresholds_mature_train_only','operational_freeze_verified','scientific_freeze_verified','user_candidate_selection','future_exact_benchmark_binding_qualified']
def branch_readiness(flags):
    missing=[k for k in PREDICTION_REQUIREMENTS if flags.get(k) is not True]
    return {'prediction_state':'PREPARED_FOR_EXTERNAL_VERIFIED_FREEZE' if not missing else 'BLOCKED_EVIDENCE','prediction_missing':missing,'economics_state':'PREPARED' if flags.get('all_P1_sessions_qualified') is True else 'BLOCKED_P1_EVIDENCE','P1_block_does_not_block_P0':True,'actual_forecasts_allowed_in_v35':False,'schedule':None,'future_dates_observed':0}
def validate_prepared_rows(rows,freeze,plan,now_at,**kw):
    # UTC-normalized identity prevents Z/+00:00 spelling from hiding duplicates.
    try:ids=[(clock(r['decision_at']),r['asset'],r['seed']) for r in rows]
    except (KeyError,ValueError,TypeError):return {'state':'BLOCKED','reason':'Malformed aware identity','actual_future_evidence':False,'qualified_dates':0}
    if len(set(ids))!=len(ids):return {'state':'BLOCKED','reason':'Duplicate semantic UTC forecast identity','actual_future_evidence':False,'qualified_dates':0}
    return validate_forward(rows,freeze,plan,now_at,**kw)
