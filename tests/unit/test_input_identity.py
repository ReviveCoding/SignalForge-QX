from copy import deepcopy
import numpy as np
import pandas as pd
from signalforge.input_identity import canonical_data_id,snapshot_inputs


def test_cache_data_identity_includes_parsed_values_clocks_and_contexts():
    acquisition={'records':[{'state':'SUCCEEDED','raw':{'sha256':'a'*64}}]}
    frame=pd.DataFrame([{'decision_time':'2020-01-01T00:00Z','y':1.,'max_dependency_available_at':'2019-12-31T00:00Z'}]);x=np.ones((1,2,3))
    original=canonical_data_id(acquisition,frame,x)
    for field,value in [('y',2.),('max_dependency_available_at','2019-12-30T00:00Z')]:
        changed=frame.copy();changed[field]=value
        assert canonical_data_id(acquisition,changed,x)!=original
    changed=x.copy();changed[0,0,0]=2.
    assert canonical_data_id(acquisition,frame,changed)!=original
    assert canonical_data_id(acquisition,frame.copy(),x.copy())==original


def test_same_input_snapshot_retains_original_receipt_across_preparation_versions(tmp_path):
    from signalforge.runtime import validate_bundle,digest
    acquisition={'records':[{'state':'SUCCEEDED','raw':{'sha256':'a'*64}}]}
    frame=pd.DataFrame([{'decision_time':'2020-01-01T00:00Z','y':1.}]);x=np.ones((1,2,3))
    identity=snapshot_inputs(tmp_path,acquisition,frame,x,'version-a')
    original=validate_bundle(tmp_path/'artifacts/model_input_snapshots'/identity)
    assert snapshot_inputs(tmp_path,acquisition,frame,x,'version-b')==identity
    assert validate_bundle(tmp_path/'artifacts/model_input_snapshots'/identity)==original
    assert original['metadata']['training_id']=='version-a'
    assert len(list((tmp_path/'artifacts/model_input_lineage').glob('*/receipt.json')))==2
