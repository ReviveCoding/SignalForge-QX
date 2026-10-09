import pytest
from signalforge.contamination import assert_unused_period,record_inspection,history_records
from signalforge.runtime import atomic_json,file_hash,validate_bundle


def test_previously_inspected_dates_remain_consumed_after_split_rename(tmp_path):
    artifact=tmp_path/'evidence.json';atomic_json(artifact,{'scope':'synthetic old development output'})
    record={'start':'2020-01-01T00:00Z','end':'2023-01-01T00:00Z','purpose':'development output inspection',
        'evidence_path':str(artifact),'evidence_sha256':file_hash(artifact),'scientific_partition':'development'}
    identity=record_inspection(tmp_path,record);receipt=validate_bundle(tmp_path/'artifacts/inspection_history'/identity)
    with pytest.raises(PermissionError,match='renaming'):assert_unused_period([record],'2021-01-01T00:00Z','2022-01-01T00:00Z')
    assert assert_unused_period([record],'2023-01-01T00:00Z','2024-01-01T00:00Z')
    atomic_json(artifact,{'changed':True})
    with pytest.raises(ValueError,match='checksum'):record_inspection(tmp_path,record)
    assert validate_bundle(tmp_path/'artifacts/inspection_history'/identity)==receipt
    history=history_records(tmp_path)
    with pytest.raises(PermissionError):assert_unused_period(history,'2022-01-01T00:00Z','2025-01-01T00:00Z')
