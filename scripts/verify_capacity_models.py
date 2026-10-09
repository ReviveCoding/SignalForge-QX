"""Actual CUDA reload and matched date diagnostics for the capacity controls."""
import json
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,validate_bundle,digest,now
from signalforge.auxiliary import sequences
from signalforge.data import auxiliary_rows
from signalforge.inference import infer_bundle
from signalforge.metrics import quantile_loss
from signalforge.models import QUANTILES
from signalforge.resources import research_gpu_stage
from signalforge.statistics import calendar_indices,paired_statistics
repo,runtime=paths();document=json.loads((repo/'reports/auxiliary_capacity_results.json').read_text())
if document.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: complete capacity comparison required')
a=json.loads((repo/'reports/eia_development_acquisition.json').read_text());frame,x=sequences(auxiliary_rows(a['records']))
index={pd.Timestamp(date).isoformat():i for i,date in enumerate(frame.decision_time)};checks=[]
losses={f:{'reference':{},'control':{}} for f in document['protocol']['families']}
with research_gpu_stage(repo,runtime,'capacity_outer_reload',600):
    for family in document['protocol']['families']:
        for year in document['protocol']['outer_years']:
            control=[];reference=[]
            for row in [r for r in document['results'] if r['family']==family and r['outer_year']==year]:
                directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
                saved=json.loads((directory/'predictions.json').read_text());raw=x[[index[pd.Timestamp(d).isoformat()] for d in saved['decision_time']]]
                mu,q=infer_bundle(directory,raw)
                delta=max(float(np.max(np.abs(mu-np.asarray(saved['mean'])))),float(np.max(np.abs(q-np.asarray(saved['quantiles'])))))
                if delta>2e-6:raise ValueError('Capacity model reload parity failure')
                ref=runtime/'artifacts/auxiliary'/row['reference_run_id'];validate_bundle(ref)
                target=json.loads((ref/'predictions.json').read_text());metric=json.loads((ref/'metrics.json').read_text())
                if pd.to_datetime(target['decision_time'],utc=True).tolist()!=pd.to_datetime(saved['decision_time'],utc=True).tolist() or target['y']!=saved['y'] or metric['normalizer_id']!=row['normalizer_id']:
                    raise ValueError('Capacity reference information/grid/target/normalizer mismatch')
                if abs(row['parameter_count']-metric['parameter_count'])/metric['parameter_count']>.01:raise ValueError('Selected control and target are not capacity matched')
                control.append(q);reference.append(np.asarray(target['quantiles']))
                checks.append({'family':family,'outer_year':year,'seed':row['seed'],'run_id':row['run_id'],'receipt_id':digest(receipt),'max_abs_reload_delta':delta,
                    'control_parameters':row['parameter_count'],'reference_parameters':metric['parameter_count'],'relative_parameter_gap':abs(row['parameter_count']-metric['parameter_count'])/metric['parameter_count']})
            y=np.asarray(saved['y'])
            for name,predictions in [('control',control),('reference',reference)]:
                values=quantile_loss(y,np.mean(predictions,axis=0),QUANTILES,row['scale'])
                for date,value in zip(saved['decision_time'],values):losses[family][name][date]=float(value)
table=[]
for family,loss in losses.items():
    dates=sorted(loss['reference']);assert dates==sorted(loss['control'])
    indices,metadata=calendar_indices(dates,8,2000)
    left=np.asarray([loss['reference'][d] for d in dates]);right=np.asarray([loss['control'][d] for d in dates])
    table.append({'family':family,'n_dates':len(dates),'reference_RGMF_pinball':float(left.mean()),'capacity_concat_pinball':float(right.mean()),
                  'paired_RGMF_minus_concat':paired_statistics(left,right,indices,metadata),'claim_scope':'Exploratory Tier B capacity diagnostic; no final selection'})
result={'state':'SUCCEEDED_ACTUAL_CUDA_CAPACITY_RELOAD_AND_COMPARISON','checks':checks,'comparison_table':table,'actual_CUDA_used':True,
        'n_verified_outer_bundles':len(checks),'qualified_for_final':False,'reserved_access':False,'at':now()}
atomic_json(repo/'reports/auxiliary_capacity_qualification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
