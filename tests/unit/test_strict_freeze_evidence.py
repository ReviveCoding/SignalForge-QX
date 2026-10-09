import json
import pytest
from signalforge.integrity import verify_frozen_models
from signalforge.runtime import atomic_json,commit_bundle,digest,file_hash


def test_frozen_family_and_original_training_receipts_cannot_be_flag_only(tmp_path):
    source={'tracks':{'Main-A':{'events':[]}}};source_path=tmp_path/'source.json';atomic_json(source_path,source)
    models=tmp_path/'models.json';atomic_json(models,{'candidates':[]})
    receipt={'components':{'models':{'path':str(models)},'source_snapshot':{'path':str(source_path)}},
        'registered_candidate_ids':['fixed'],'tracks':{'Main-A':{}}}
    with pytest.raises(PermissionError,match='full frozen'):verify_frozen_models(receipt,tmp_path)
    bundle=tmp_path/'artifacts/models'/'one'
    member_receipt=commit_bundle(bundle,{'model.bin':b'fixture only'},{'evidence_kind':'synthetic_fixture'})
    candidate={'candidate_id':'fixed','track':'Main-A','members':[{'relative_bundle':'artifacts/models/one','receipt_id':digest(member_receipt)}]}
    atomic_json(models,{'candidates':[candidate]})
    with pytest.raises(PermissionError,match='strict original-source'):verify_frozen_models(receipt,tmp_path)
    candidate['members'][0]['relative_bundle']='../unrelated'
    atomic_json(models,{'candidates':[candidate]})
    with pytest.raises(PermissionError,match='outside'):verify_frozen_models(receipt,tmp_path)
