from qualification_v35.forward_contract_v2 import *
def test_P1_block_is_independent():
    flags={k:True for k in PREDICTION_REQUIREMENTS};flags['all_P1_sessions_qualified']=False;r=branch_readiness(flags);assert r['prediction_state']=='PREPARED_FOR_EXTERNAL_VERIFIED_FREEZE';assert r['economics_state']=='BLOCKED_P1_EVIDENCE' and not r['actual_forecasts_allowed_in_v35']
def test_missing_future_benchmark_binding():
    flags={k:True for k in PREDICTION_REQUIREMENTS};flags['future_exact_benchmark_binding_qualified']=False;assert 'future_exact_benchmark_binding_qualified' in branch_readiness(flags)['prediction_missing']
def test_timezone_duplicate_spelling_blocked():
    rows=[{'decision_at':'2030-01-04T23:00:00Z','asset':'IEF','seed':11},{'decision_at':'2030-01-04T23:00:00+00:00','asset':'IEF','seed':11}];assert validate_prepared_rows(rows,None,{},'2031-01-01T00:00:00Z')['reason']=='Duplicate semantic UTC forecast identity'
def test_missing_freeze_no_access():
    r=validate_prepared_rows([],None,{},'2031-01-01T00:00:00Z');assert r['state']=='BLOCKED' and not r['actual_future_evidence']
