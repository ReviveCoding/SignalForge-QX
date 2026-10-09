import numpy as np
import pandas as pd
import pytest
from signalforge.statistics import calendar_indices,paired_statistics,exploratory_fdr


def test_seeds_do_not_add_dates_and_underpowered_null_is_not_equivalence():
    dates=pd.date_range('2020-01-03',periods=16,freq='7D',tz='UTC')
    draws,metadata=calendar_indices(dates,8,2000)
    result=paired_statistics(np.ones(16),np.ones(16),draws,metadata)
    assert result['n_dates']==16 and result['underpowered']
    assert result['two_sided_centered_block_p']==1 and result['equivalence_established'] is False
    assert result['n_training_seeds_are_independent_market_samples'] is False
    with pytest.raises(ValueError,match='Unique ordered'):calendar_indices(list(dates)*3,8,2000)


def test_exploratory_fdr_full_family_missing_and_dependence_sensitivity():
    result=exploratory_fdr({'a':.01,'b':.04,'missing':None})
    assert result['BH']['a']==pytest.approx(.03) and result['BH']['b']==pytest.approx(.06)
    assert result['BH']['missing'] is None and result['family_size']==3
    assert result['BY_arbitrary_dependence']['a']>result['BH']['a']
    reversed_=exploratory_fdr({'missing':None,'b':.04,'a':.01})
    assert reversed_['BH']==result['BH']
    with pytest.raises(ValueError):exploratory_fdr({'bad':float('nan')})
