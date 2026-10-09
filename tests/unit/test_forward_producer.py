import pytest
from signalforge.forward import validate_forward,produce_once


def test_forward_issuance_uses_actual_clock_and_no_outcomes(tmp_path):
    data={'decision_time':'2026-10-09T22:00Z','expected_outcome_available_at':'2026-10-16T22:00Z','max_dependency_available_at':'2026-10-09T21:00Z'}
    assert validate_forward(data,'2026-10-08T22:00Z','2026-10-09T22:01Z')
    with pytest.raises(PermissionError):validate_forward(data,'2026-10-08T22:00Z','2026-10-10T22:00Z')
    with pytest.raises(PermissionError):validate_forward({**data,'y':0},'2026-10-08T22:00Z','2026-10-09T22:01Z')
    with pytest.raises(RuntimeError,match='operational freeze'):produce_once(tmp_path,tmp_path,'NOW')
