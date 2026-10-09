import copy
import pandas as pd
import pytest
from signalforge.forward_diagnostics import zero_shot_preprocessing,admission

def test_unseen_group_labels_cannot_fit_embedding_or_normalizer():
    frame=pd.DataFrame({'asset':['SPY','SPY','QQQ'],'decision_time':['2022-01-01T00:00Z']*3,'label_available_at':['2022-01-08T00:00Z']*3,'x':[[1.],[3.],[999.]],'y':[.1,.3,'unreadable held-out label']})
    first=zero_shot_preprocessing(frame,{'QQQ'},'2022-02-01T00:00Z')
    frame.loc[2,'y']='different forbidden target';frame.at[2,'x']=[-999999.]
    assert first==zero_shot_preprocessing(frame,{'QQQ'},'2022-02-01T00:00Z')
    assert first['training_groups']==['SPY'] and first['shared_scale']==pytest.approx(.1)
    with pytest.raises(ValueError):zero_shot_preprocessing(frame,{'QQQ','SPY'},'2022-02-01T00:00Z')

def test_future_admission_cannot_bypass_missing_science_authorization_or_bill():
    plan={'budget':{'requested_ceiling_gpu_seconds':100}}
    evidence={k:True for k in ['explicit_separate_run_authorization','current_full_validation','measured_pilot_bound_to_plan','original_source_clock_qualified','operational_freeze_verified','new_cohort_unseen','future_cohort_available']}
    evidence.update(conservative_gpu_seconds=50,use_reserved_data=False)
    assert admission(plan,evidence)['state']=='ADMITTED'
    for key in list(evidence):
        wrong=copy.deepcopy(evidence);wrong[key]=101 if key=='conservative_gpu_seconds' else True if key=='use_reserved_data' else False
        assert admission(plan,wrong)['missing_dependencies']

def test_forward_transport_counts_all_eight_physical_assets_in_both_tracks():
    import json
    from pathlib import Path
    from signalforge.forward_diagnostics import planned_fit_counts
    plan=json.loads((Path(__file__).resolve().parents[2]/'configs/forward_diagnostic_protocol_v32.json').read_text())
    counts=planned_fit_counts(plan)
    assert counts=={'matched_primary':184,'retrained_components':552,'ssl_paired':184,'zero_shot_physical_groups':1472,'total':2392}
    assert len(plan['hpo']['rgmf_gru']['width'])*len(plan['hpo']['rgmf_gru']['lr'])==20
    assert plan['budget']['authorized_additional_gpu_seconds']==0
