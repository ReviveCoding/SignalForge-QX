import pytest
from signalforge.research_cards import validate_future_cards


def test_future_version_cannot_reuse_consumed_final_or_omit_failure_criterion():
    card={k:'registered evidence' for k in ['observed_limitation','new_hypothesis','needed_data','baseline','experiment','budget','success_failure_criterion','claim_boundary','new_protocol']}
    card.update(cohort_policy='new_unused_or_post_actual_freeze_prospective',reuse_consumed_final=False)
    assert validate_future_cards([card])
    card['reuse_consumed_final']=True
    with pytest.raises(PermissionError):validate_future_cards([card])
    card['reuse_consumed_final']=False;card.pop('success_failure_criterion')
    with pytest.raises(ValueError):validate_future_cards([card])
