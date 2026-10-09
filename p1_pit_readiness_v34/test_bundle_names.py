from pathlib import Path
import tempfile,pytest
from signalforge.runtime import commit_bundle,validate_bundle,file_hash
from p1_pit_readiness_v34.bundle_names import flatten
T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/tmp')
def test_runtime_contract_rejects_paths():
    with tempfile.TemporaryDirectory(dir=T) as tmp:
        with pytest.raises(ValueError,match='Invalid artifact name'):commit_bundle(Path(tmp)/'b',{'reports/a.txt':b'a'},{})
def test_flat_names_commit_actual_bundle():
    with tempfile.TemporaryDirectory(dir=T) as tmp:
        values,mapping=flatten({'reports/a.txt':b'a'});b=Path(tmp)/'b';commit_bundle(b,values|{'file_map.json':mapping},{});validate_bundle(b);assert (b/next(iter(values))).read_bytes()==b'a'
def test_equal_basenames_do_not_collide():
    f,m=flatten({'reports/a/tests.json':b'a','reports/b/tests.json':b'b'});assert len(f)==2 and len(m)==2 and all(Path(k).name==k for k in f)
def test_no_byte_mutation():
    f,m=flatten({'reports/calibration_v34/model.csv':b'original bytes'});assert list(f.values())==[b'original bytes'] and list(m.values())==['reports/calibration_v34/model.csv']