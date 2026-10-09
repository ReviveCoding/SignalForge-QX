"""Track CUDA planning blockers are separate from usable CPU/source branches."""
import json
from .runtime import file_hash

def track_cuda_pilot_boundary(repo):
    profile=json.loads((repo/'configs/local_rtx4090_laptop.json').read_text())
    ceiling=profile['budget']['pilot_completed_fit_ceiling'];seconds=profile['budget']['pilot_gpu_seconds']
    evidence={};completed=0
    capacity=repo/'reports/auxiliary_capacity_bill.json'
    auxiliary=repo/'reports/auxiliary_pilot_costs.json'
    if capacity.is_file():
        bill=json.loads(capacity.read_text());completed=bill['cumulative_pilot_completed']
        evidence['capacity_bill']={'path':str(capacity),'sha256':file_hash(capacity)}
    elif auxiliary.is_file():
        costs=json.loads(auxiliary.read_text());completed=sum(r['successful_pilot_fits'] for r in costs['families'].values())
        evidence['auxiliary_pilots']={'path':str(auxiliary),'sha256':file_hash(auxiliary)}
    if type(completed) is not int or completed<0 or completed>ceiling:raise ValueError('Cumulative pilot accounting requires reconciliation')
    return {'state':'PAUSED_PILOT_FIT_CEILING' if completed==ceiling else 'BLOCKED_REQUIRED_MEASURED_PILOT',
        'cumulative_completed_pilot_fits':completed,'completed_fit_ceiling':ceiling,'remaining_fit_slots':ceiling-completed,
        'pilot_seconds_ceiling':seconds,'actual_pilot_evidence':evidence,'new_pilot_fits':0,'CUDA_training_executed':False,
        'reason':'Main/Nested actual workload pilot and admitted complete bill are absent. Auxiliary timing is not silently transferred to different source/asset/shape workloads. Current fit/seconds ceilings cannot be automatically increased.',
        'budget_ceiling_changed':False,'qualified_for_final':False}
