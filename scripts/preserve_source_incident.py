import json
import numpy as np
from signalforge.runtime import paths,commit_bundle,digest,now
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences
repo,runtime=paths();acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
frame,x=sequences(auxiliary_rows(acquisition['records']))
commit_bundle(runtime/'artifacts/incidents/eia_2019_duplicate_original_inputs',{'acquisition.json':acquisition,'frame.json':frame.to_dict('records'),'sequences.json':x.tolist()}, {'state':'INVALID_SOURCE_PRESERVED_FOR_EXACT_INPUT_AUDIT','at':now(),'reserved_access':False})
print('Original acquisition and full development inputs preserved',len(frame),x.shape)
