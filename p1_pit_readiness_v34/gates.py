"""Fail-closed evidence gates; no training, trades, reserved access or clock invention."""
from datetime import datetime,timedelta,timezone

def certify(version,proof):
    needed=['payload_sha256','first_public_at','publisher','dissemination_record','version_link','correction_history_complete']
    if not all(proof.get(k) for k in needed):return False
    if proof.get('clock_kind')!='authenticated_first_public':return False
    if proof['payload_sha256']!=version['payload_sha256']:return False
    return True

def retrieval_allowed(status):return status not in ['HTTP_403_ENTITLEMENT','BLOCKED_AUTH','RESERVED','ACCESS_DENIED']
def p1_gate(flags):return all(flags.get(k) is True for k in ['raw_open','open_clock','actions_complete','ex_pay_clocks','split_share_units','cash_asof','license','accounting_oracles'])
def future_probe(*,operational_freeze=None,model_bound=False,clock_qualified=False,cohort_dates=0,independent_reviewer=False):
    qualified=bool(operational_freeze and model_bound and clock_qualified and cohort_dates>=104 and independent_reviewer)
    schedule=None
    if operational_freeze:
        t=datetime.fromisoformat(operational_freeze.replace('Z','+00:00'))
        if t.tzinfo is None:raise ValueError('Aware actual operational freeze')
        schedule={'first_52_week_minimum':(t+timedelta(weeks=52)).isoformat(),'two_52_week_minimum':(t+timedelta(weeks=104)).isoformat(),'additional_label_maturity_required':True,'not_guaranteed_availability':True}
    return {'state':'READY' if qualified else 'BLOCKED_DATA_OR_FUTURE','operational_freeze_attested':bool(operational_freeze),'model_binding':model_bound,'strict_clock_qualified':clock_qualified,'actual_new_mature_dates':cohort_dates,'independent_reviewer':independent_reviewer,'schedule':schedule,'reserved_access':False,'forecasts_issued':0}
def final_gate(authorization,freeze,strict_pit):return bool(authorization and freeze and strict_pit)