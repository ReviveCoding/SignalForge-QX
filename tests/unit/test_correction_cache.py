import numpy as np
import pandas as pd
from signalforge.correction_cache import partition_signature,numeric_dependencies


def frame():
    return pd.DataFrame([{'decision_time':'2018-01-05T23:00Z','label_start':'2018-01-05T23:00Z','label_end':'2018-01-11T05:00Z','label_available_at':'2018-01-11T05:00Z','max_dependency_available_at':'2018-01-04T05:00Z','y':1.,'sequence_index':0}])


def test_partition_proof_includes_values_targets_and_clocks():
    train=frame();test=frame();x=np.ones((1,26,14));signature=partition_signature(train,test,x)
    for field,value in [('y',2.),('label_available_at','2018-01-12T05:00Z'),('max_dependency_available_at','2018-01-05T05:00Z')]:
        changed=test.copy();changed[field]=value
        assert partition_signature(train,changed,x)!=signature
    changed=x.copy();changed[0,0,0]=2.
    assert partition_signature(train,test,changed)!=signature


def test_only_source_preparation_can_change_with_exact_input_proof():
    deps={'files':{'data':'old','sources':'old','models':'same','features':'same'},'environment_id':'same','functions':{'fit':'same'}}
    changed={'files':{**deps['files'],'data':'new','sources':'new'},'environment_id':'same','functions':{'fit':'same'}}
    assert numeric_dependencies(deps)==numeric_dependencies(changed)
    changed['files']['models']='new'
    assert numeric_dependencies(deps)!=numeric_dependencies(changed)
    assert deps['files']['data']=='old'
