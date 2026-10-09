import pytest
from signalforge.capacity import CapacityMatchedCUDA


@pytest.mark.parametrize('kind',['gru','transformer'])
@pytest.mark.parametrize('width',[8,16,32,48,64])
def test_outcome_independent_capacity_match_actual_parameter_counts(kind,width):
    adapter=CapacityMatchedCUDA(kind,width=width)
    model=adapter.build(28);spec=adapter.capacity_spec
    assert sum(p.numel() for p in model.parameters())==spec['control_parameters']
    assert spec['relative_parameter_gap']<=.01 and spec['control_parameters']<=500000
    with pytest.raises(ValueError,match='schema'):adapter.build(30)
