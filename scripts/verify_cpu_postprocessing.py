import json,sys
from contextlib import nullcontext
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,validate_bundle,digest,now
from signalforge.data import auxiliary_rows
from signalforge.inference import infer_bundle
from signalforge.frozen_processing import validate_frozen_postprocessing
from signalforge.resources import research_gpu_stage
from signalforge.development import CPU_FAMILIES
repo,runtime=paths();all_families='--all' in sys.argv
prefix='auxiliary_all_family_postprocessing' if all_families else 'auxiliary_cpu_postprocessing'
doc=json.loads((repo/'reports'/(prefix+'.json')).read_text())
validate_frozen_postprocessing(doc['candidates'],doc['ensemble'],doc['calibration'])
a=json.loads((repo/'reports/eia_development_acquisition.json').read_text());frame=auxiliary_rows(a['records']);contexts={}
for i in range(25,len(frame)):
    row=frame.iloc[i]
    if '2023-01-01'<=row.decision_time<'2024-01-01':contexts[pd.Timestamp(row.decision_time).isoformat()]=np.array(frame.iloc[i-25:i+1].x.tolist())
checks=[]
with research_gpu_stage(repo,runtime,'all_family_finalist_reload',600) if all_families else nullcontext():
 for candidate in doc['candidates']:
    for member in candidate['members']:
        directory=runtime/member['relative_bundle'];receipt=validate_bundle(directory);assert digest(receipt)==member['receipt_id']
        saved=json.loads((directory/'predictions.json').read_text());x=np.array([contexts[pd.Timestamp(date).isoformat()] for date in saved['decision_time']])
        mu,q=infer_bundle(directory,x);delta=max(float(np.max(np.abs(mu-np.array(saved['mean'])))),float(np.max(np.abs(q-np.array(saved['quantiles'])))))
        family=candidate['candidate_id'].removeprefix('aux_cpu_').removeprefix('aux_all_')
        tolerance=1e-10 if family in CPU_FAMILIES|{'lightgbm','xgboost','lgb_no_age','lgb_no_coverage'} else 2e-6
        assert delta<=tolerance
        checks.append({'candidate_id':candidate['candidate_id'],'member_id':member['member_id'],'receipt_id':member['receipt_id'],'max_abs_delta':delta})
result={'state':'VERIFIED_REAL_ALL_FAMILY_POSTPROCESSING_RELOAD_AND_SCHEMA' if all_families else 'VERIFIED_REAL_CPU_POSTPROCESSING_RELOAD_AND_SCHEMA','checks':checks,'n_bundles':len(checks),'n_forecast_dates':len(contexts),
        'new_model_fits':doc['new_model_fits'],'verified_reused_model_fits':doc['verified_reused_model_fits'],
        'qualified_for_final':False,'source_gaps_retained':True,'final_access':False,'at':now()}
atomic_json(repo/'reports'/(prefix+'_qualification.json'),result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
