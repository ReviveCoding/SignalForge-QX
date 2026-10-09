import copy,json
import pytest
from signalforge.runtime import commit_bundle,file_hash
from signalforge.forward import append_mature_outcome


def test_mature_forward_outcome_append_only_grid_clocks_and_versions(tmp_path):
    directory=tmp_path/'artifacts/forward'/'fixed'
    forecast={'freeze_id':'one-version','issued_at':'2026-10-02T22:02Z','decision_time':'2026-10-02T22:00Z','assets':['a','b']}
    commit_bundle(directory,{'forecast.json':forecast},{'freeze_id':'one-version','prospective':True})
    checksum=file_hash(directory/'forecast.json')
    outcome={'decision_time':forecast['decision_time'],'assets':['a','b'],'values':[.01,-.02],
        'label_end':'2026-10-09T22:00Z','available_at':'2026-10-10T00:00Z','raw_hashes':['a'*64]}
    with pytest.raises(PermissionError,match='mature'):append_mature_outcome(tmp_path,directory,outcome,'2026-10-09T23:00Z')
    result=append_mature_outcome(tmp_path,directory,outcome,'2026-10-10T01:00Z')
    assert result==append_mature_outcome(tmp_path,directory,outcome,'2026-10-10T02:00Z')
    changed=copy.deepcopy(outcome);changed['values'][0]=.02
    with pytest.raises(PermissionError,match='Append-only'):append_mature_outcome(tmp_path,directory,changed,'2026-10-10T02:00Z')
    changed=copy.deepcopy(outcome);changed['assets'].reverse()
    with pytest.raises(ValueError,match='grid'):append_mature_outcome(tmp_path,directory,changed,'2026-10-10T02:00Z')
    changed=copy.deepcopy(outcome);changed['label_end']='2026-10-02T21:59Z'
    with pytest.raises(PermissionError,match='issued first'):append_mature_outcome(tmp_path,directory,changed,'2026-10-10T02:00Z')
    assert file_hash(directory/'forecast.json')==checksum
