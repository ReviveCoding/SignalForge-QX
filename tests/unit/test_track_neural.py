import numpy as np
import pytest
from signalforge.track_neural import raw_source_validity,ExplicitSourceCUDA


def test_source_validity_excludes_finite_age_coverage_and_imputed_zero():
    raw=np.array([[[np.nan,np.nan,7.,0.]],[[0.,np.nan,7.,1.]]])
    assert raw_source_validity(raw,[[0,1]]).tolist()==[[False],[True]]
    assert raw_source_validity(raw,[]).shape==(2,0)
    with pytest.raises(ValueError):raw_source_validity(raw,[[8]])
    with pytest.raises(ValueError):raw_source_validity(raw,[[]])


def test_rgfm_training_never_implicitly_uses_age_masks_before_cuda_allocation():
    adapter=ExplicitSourceCUDA('rgmf_gru',base_columns=[0],source_columns=[[1,2]],meta_columns=[3])
    with pytest.raises(ValueError,match='Explicit typed raw source'):
        adapter.fit(np.ones((2,3,4)),np.ones(2),1.,observed=np.ones((2,3,4),dtype=bool))
    with pytest.raises(ValueError,match='Explicit raw-value'):
        adapter.forward(None)
