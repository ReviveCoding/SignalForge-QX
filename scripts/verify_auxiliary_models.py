"""Verify stored real forecasts against freshly loaded qualified local models."""
import json,sys
from contextlib import nullcontext
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,validate_bundle,now,digest,gpu_lease
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences
from signalforge.development import CPU_FAMILIES
from signalforge.inference import infer_bundle
from signalforge.resources import research_gpu_stage

repo,runtime=paths()
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
frame,x=sequences(auxiliary_rows(acquisition['records']))
index={pd.Timestamp(date).isoformat():i for i,date in enumerate(frame.decision_time)}
document=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
all_families='--all' in sys.argv
if all_families and document.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: complete all-family development required')
checks=[];failures=[]
with research_gpu_stage(repo,runtime,'all_family_outer_reload',600) if all_families else nullcontext():
    for row in document['results']:
        if row.get('state')!='SUCCEEDED' or (not all_families and row['family'] not in CPU_FAMILIES):continue
        try:
            directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
            stored=json.loads((directory/'predictions.json').read_text())
            raw=x[[index[pd.Timestamp(date).isoformat()] for date in stored['decision_time']]]
            mean,q=infer_bundle(directory,raw)
            dm=float(np.max(np.abs(mean-np.asarray(stored['mean']))));dq=float(np.max(np.abs(q-np.asarray(stored['quantiles']))))
            tolerance=1e-10 if row['family'] in CPU_FAMILIES|{'lightgbm','xgboost','lgb_no_age','lgb_no_coverage'} else 2e-6
            if dm>tolerance or dq>tolerance:raise ValueError(f'Real model/forecast reload parity failure: mean={dm}, quantiles={dq}, tolerance={tolerance}')
            checks.append({'run_id':row['run_id'],'receipt_id':digest(receipt),'family':row['family'],
                           'max_mean_abs_delta':dm,'max_quantile_abs_delta':dq,'tolerance':tolerance,'n_dates':len(raw)})
        except Exception as error:failures.append({'run_id':row['run_id'],'family':row['family'],'failure_type':type(error).__name__,'reason':str(error)})
result={'created_at':now(),'state':'FAILED_RELOAD' if failures else 'SUCCEEDED_ALL_FAMILY_RELOAD' if all_families else 'SUCCEEDED_CPU_RELOAD','checks':checks,'failures':failures,'n_verified_outer_bundles':len(checks),
        'research_scope':'Tier B physical-only development; original-vintage/final/economic qualification remains separate',
        'actual_CUDA_used':all_families,'final_access':False}
atomic_json(repo/'reports'/('auxiliary_all_family_reload_qualification.json' if all_families else 'auxiliary_real_reload_qualification.json'),result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
raise SystemExit(1 if failures else 0)
