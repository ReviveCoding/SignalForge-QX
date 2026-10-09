from pathlib import Path
import pytest
from signalforge.boundary import training_import_boundary


def test_training_graph_has_no_reserved_loader():
    root=Path(__file__).parents[2]/'src/signalforge'
    audit=training_import_boundary(root,['development','models','features','ssl','residual'])
    assert 'final' not in audit['modules'] and not audit['reserved_access']


def test_transitive_and_conditional_reserved_import_rejected(tmp_path):
    (tmp_path/'train.py').write_text('from .helper import f')
    (tmp_path/'helper.py').write_text('def f():\n    from .final import read_reserved\n')
    (tmp_path/'final.py').write_text('def read_reserved(): pass')
    with pytest.raises(PermissionError,match='reserved'):training_import_boundary(tmp_path,['train'])
