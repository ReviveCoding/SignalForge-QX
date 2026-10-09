"""Future research must be traceable to limitations and use a new cohort."""
def validate_future_cards(cards):
    required={'observed_limitation','new_hypothesis','needed_data','baseline','experiment','budget',
              'success_failure_criterion','claim_boundary','new_protocol','cohort_policy'}
    if not isinstance(cards,list) or not cards:raise ValueError('Nonempty future research cards required')
    for card in cards:
        if not required<=set(card) or any(not isinstance(card[k],str) or not card[k].strip() for k in required):
            raise ValueError('Complete limitation-to-experiment evidence chain required')
        if card.get('reuse_consumed_final') is not False or card['cohort_policy']!='new_unused_or_post_actual_freeze_prospective':
            raise PermissionError('Future method cannot retune a consumed final cohort')
    return True
