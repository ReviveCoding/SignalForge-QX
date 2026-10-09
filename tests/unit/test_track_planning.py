import pytest
from signalforge.runtime import atomic_json
from signalforge.track_planning import track_cuda_pilot_boundary

def setup(tmp_path):
    atomic_json(tmp_path/'configs/local_rtx4090_laptop.json',{'budget':{'pilot_completed_fit_ceiling':120,'pilot_gpu_seconds':3600}})

def test_missing_track_workload_pilot_is_not_auxiliary_cost_transfer(tmp_path):
    setup(tmp_path);result=track_cuda_pilot_boundary(tmp_path)
    assert result['state']=='BLOCKED_REQUIRED_MEASURED_PILOT' and result['remaining_fit_slots']==120
    assert result['new_pilot_fits']==0 and not result['CUDA_training_executed'] and not result['qualified_for_final']

def test_cumulative_pilot_ceiling_blocks_without_reset_or_budget_increase(tmp_path):
    setup(tmp_path);atomic_json(tmp_path/'reports/auxiliary_capacity_bill.json',{'cumulative_pilot_completed':120})
    result=track_cuda_pilot_boundary(tmp_path)
    assert result['state']=='PAUSED_PILOT_FIT_CEILING' and result['remaining_fit_slots']==0
    assert result['completed_fit_ceiling']==120 and result['pilot_seconds_ceiling']==3600 and not result['budget_ceiling_changed']

def test_auxiliary_only_accounting_and_impossible_count_require_reconciliation(tmp_path):
    setup(tmp_path);atomic_json(tmp_path/'reports/auxiliary_pilot_costs.json',{'families':{'one':{'successful_pilot_fits':6},'two':{'successful_pilot_fits':6}}})
    assert track_cuda_pilot_boundary(tmp_path)['cumulative_completed_pilot_fits']==12
    atomic_json(tmp_path/'reports/auxiliary_capacity_bill.json',{'cumulative_pilot_completed':121})
    with pytest.raises(ValueError,match='reconciliation'):track_cuda_pilot_boundary(tmp_path)
