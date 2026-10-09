import json,pandas as pd
from signalforge.eia_market_integration import FIELD_MAP


def test_eia_market_registry_is_uso_only_and_has_no_gas_invention():
    assert set(FIELD_MAP.values())=={'commercial_crude_value','commercial_crude_change','cushing_value','cushing_change','gasoline_change','distillate_change'}
    assert not any('gas' in field and field!='gasoline_change' for field in FIELD_MAP.values())


def test_source_feature_registry_marks_eia_not_applicable_instead_of_zero():
    from pathlib import Path
    repo=Path(__file__).parents[2];registry=json.loads((repo/'configs/source_feature_registry_v31.json').read_text())
    assert registry['source_applicability']['eia']['USO'] is True
    assert all(not v for k,v in registry['source_applicability']['eia'].items() if k!='USO')
    assert 'no zeros' in registry['claim_boundary']
