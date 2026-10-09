"""Canonical parsed inputs participate in model cache identity, not only raw bytes."""
import json
import numpy as np
import pandas as pd
from .runtime import digest,commit_bundle,validate_bundle


def canonical_data_id(acquisition,frame,x):
    return digest({'schema':'canonical_model_inputs_v1','raw':[r['raw']['sha256'] for r in acquisition['records'] if r['state']=='SUCCEEDED'],
                   'frame':frame.to_dict('records'),'contexts':x.tolist()})


def snapshot_inputs(runtime,acquisition,frame,x,training_id):
    legacy=digest({'raw':[r['raw']['sha256'] for r in acquisition['records'] if r['state']=='SUCCEEDED'],'grid':frame.decision_time.tolist()})
    identity=canonical_data_id(acquisition,frame,x)
    directory=runtime/'artifacts/model_input_snapshots'/identity
    if (directory/'receipt.json').exists():
        receipt=validate_bundle(directory)
        stored_id=canonical_data_id(json.loads((directory/'acquisition.json').read_text()),
            pd.DataFrame(json.loads((directory/'frame.json').read_text())),np.asarray(json.loads((directory/'sequences.json').read_text())))
        if receipt['metadata'].get('canonical_data_id')!=identity or stored_id!=identity:
            raise ValueError('Canonical input snapshot identity mismatch')
    else:receipt=commit_bundle(directory,
                  {'acquisition.json':acquisition,'frame.json':frame.to_dict('records'),'sequences.json':x.tolist()},
                  {'legacy_data_id':legacy,'canonical_data_id':identity,'training_id':training_id,'final_access':False})
    # Same numerical corpus can have several preparation implementations. Retain
    # the original immutable snapshot and bind each current implementation apart.
    lineage={'canonical_data_id':identity,'training_id':training_id,'input_snapshot_receipt_id':digest(receipt),'final_access':False}
    commit_bundle(runtime/'artifacts/model_input_lineage'/digest(lineage),{'lineage.json':lineage},lineage)
    return identity
