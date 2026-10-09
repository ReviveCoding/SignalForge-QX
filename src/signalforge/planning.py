"""Evidence-driven branch planning; admission never implies research completion."""
import json,sqlite3
from .runtime import now


def run_plan(repo,runtime,mode):
    def load(name):
        path=repo/'reports'/name
        return json.loads(path.read_text()) if path.exists() else {}
    bill=load('auxiliary_measured_run_bill.json');source=load('eia_corrected_source_audit.json')
    acquisition=load('eia_development_acquisition.json');protocol=load('auxiliary_registered_protocol.json')
    executed=load('auxiliary_development_results.json');active=load('auxiliary_execution_state.json')
    state='BLOCKED_DATA';reasons=[]
    if source.get('state')!='PASSED_CORRECTED_DEVELOPMENT_SOURCE':reasons.append('Corrected source-clock/target audit absent or failed')
    if not protocol or protocol.get('final_access') is not False:reasons.append('Registered pre-reserved development protocol absent')
    if not bill or bill.get('unmeasured_families'):reasons.append('Measured family costs incomplete')
    remaining=bill.get('remaining_gpu_seconds')
    ledger=runtime/'ledger/development_compute.sqlite'
    if ledger.exists():
        connection=sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True)
        try:
            ceiling=connection.execute('SELECT ceiling FROM budget_settings WHERE id=1').fetchone()[0]
            charged=connection.execute('SELECT COALESCE(SUM(seconds),0) FROM compute_charges').fetchone()[0]
            remaining=max(0,ceiling-charged)
        finally:connection.close()
    if not reasons:
        state='READY_AUXILIARY_DEVELOPMENT' if bill.get('state')=='READY' and remaining is not None and remaining>0 else 'PAUSED_BUDGET'
    return {'created_at':now(),'mode':mode,'state':state,'reasons':reasons,'active_execution':active.get('state','NOT_RUN'),
            'nominal_registered_recipe_count':sum(bill.get('planned_fit_counts',{}).values()),
            'successful_outer_bundles':sum(r.get('state')=='SUCCEEDED' for r in executed.get('results',[]) if r.get('run_id')),
            'cumulative_gpu_seconds_remaining':remaining,'configured_gpu_seconds':protocol.get('budget_seconds',43200),
            'branch_states':{'Auxiliary-C':state,'Main-A':'BLOCKED_DATA_AUTH_ORIGINAL_CLOCKS','Nested-B':'BLOCKED_DATA_AUTH_DISSEMINATION','P1_economics':'BLOCKED_PRICE_P1','final':'SEALED_FREEZE_GATES_INCOMPLETE','forward':'BLOCKED_OPERATIONAL_FREEZE'},
            'whole_archive_complete':acquisition.get('completed')==acquisition.get('required') if acquisition else False,
            'strict_PIT_qualified':False,'authoritative_full_study':False,'reserved_access':False,
            'resume_command':".\\scripts\\Invoke-SignalForge.ps1 -Action python -Rest @('scripts/run_auxiliary_development.py')",
            'next_executable':'Continue admitted Auxiliary grid plus independent software/CPU source work; use actual GPU lease before CUDA'}
