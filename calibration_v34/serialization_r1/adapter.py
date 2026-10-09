"""v1r1 native JSON scalar adapter; identical scientific numeric operations."""
import json,sys
from pathlib import Path
import numpy as np
from signalforge.runtime import digest,file_hash,atomic_json
from calibration_v34 import runner as core
from calibration_v34.api import fit as original_fit

def native(x):
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x

def corrected_fit(*args,**kwargs):return native(original_fit(*args,**kwargs))
def identity_hash():return digest({p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
def main():
    c=core.cfg();p=Path(__file__).parent/'protocol_v1r1.json';contract={'protocol_id':'sgqx-v34-calibration-native-JSON-serialization-v1r1','parent_OOF_configuration_id':c['configuration_id'],'parent_code_hash':core.source_hash(),'revision_code_hash':identity_hash(),'repair':'numpy scalars converted to exact native JSON equivalents before hash; calibration arithmetic, candidates, selection and support unchanged','new_GPU_training':0,'reserved_access':False,'outer_access_before_refit':False};contract['revision_id']=digest(contract)
    if p.exists():assert json.loads(p.read_text())==contract
    else:atomic_json(p,contract)
    core.fit=corrected_fit;core.calibrate();cal=json.loads((core.O/'calibration_frozen.json').read_text());failure=json.loads((core.O/'failure_calibration_serialization_v1.json').read_text());assert failure['OOF_sha256']==file_hash(core.O/'genuine_sia_oof.csv');atomic_json(core.O/'serialization_repair_v1r1.json',{'revision':contract,'calibration_parameter_id':cal['parameter_id'],'calibration_receipt_sha256':file_hash(core.O/'calibration_frozen.json'),'OOF_hash_unchanged':True,'GPU_refits':0,'failed_prior_attempt_preserved':True});print(json.dumps({'serialization_revision':'v1r1','revision_id':contract['revision_id'],'calibration_parameter_id':cal['parameter_id'],'GPU_refits':0}),flush=True)
if __name__=='__main__':main()