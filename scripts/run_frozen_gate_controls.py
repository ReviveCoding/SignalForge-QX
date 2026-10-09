"""E03 frozen gate intervention, explicitly distinct from retrained ablations."""
import json
import numpy as np
import pandas as pd
import torch
from signalforge.runtime import paths,validate_bundle,digest,atomic_json,commit_bundle,now
from signalforge.resources import research_gpu_stage
from signalforge.models import NeuralCUDA,QUANTILES
from signalforge.features import TrainTransform
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences
from signalforge.statistics import calendar_indices,paired_statistics
from signalforge.metrics import quantile_loss
repo,runtime=paths();doc=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
if doc.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: complete core development required')
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text());frame,raw=sequences(auxiliary_rows(acquisition['records']))
index={pd.Timestamp(d).isoformat():i for i,d in enumerate(frame.decision_time)}
families=['rgmf_linear','rgmf_gru','rgmf_transformer'];losses={family:{v:{} for v in ['reference','uniform_valid_gate','base_only']} for family in families};checks=[]
protocol={'experiment':'E03','intervention_type':'frozen_gate_intervention','retrained':False,'families':families,
    'outer_years':[2018,2019,2020,2021,2022],'seeds':[11,37,71],
    'cases':['uniform_valid_gate','base_only'],'uniform_gate':'zero frozen gate weights/bias; preserve explicit source-validity mask',
    'base_only':'joint-trained base branch with all source validity false; not an independently retrained base model',
    'claim_scope':'Exploratory component sensitivity of stored models; no causal source importance or retrained component effect',
    'reserved_access':False,'promotion_eligible':False}
atomic_json(repo/'reports/frozen_gate_controls_protocol.json',protocol)
with research_gpu_stage(repo,runtime,'frozen_gate_controls',900):
    for family in families:
        for year in protocol['outer_years']:
            predictions={name:[] for name in losses[family]}
            for seed in protocol['seeds']:
                row=next(r for r in doc['results'] if r['family']==family and r['outer_year']==year and r['seed']==seed)
                directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
                saved=json.loads((directory/'predictions.json').read_text());metric=json.loads((directory/'metrics.json').read_text())
                x=raw[[index[pd.Timestamp(d).isoformat()] for d in saved['decision_time']]];observed=np.isfinite(x)
                manifest=json.loads((directory/'transform.json').read_text());transform=TrainTransform()
                transform.center=np.asarray(manifest['center']);transform.scale=np.asarray(manifest['scale'])
                transformed=transform.transform(x.reshape(-1,x.shape[-1])).reshape(len(x),x.shape[1],-1)
                masks=np.concatenate([observed,observed],axis=-1)
                valid=np.stack([observed[:,-1,columns].any(1) for columns in metric['raw_source_validity_columns']],1)
                model=NeuralCUDA.load(directory/'model.bin')
                mu,q=model.predict(transformed,observed=masks,elapsed=np.zeros_like(transformed),source_valid=valid)
                delta=max(float(np.max(np.abs(mu-np.asarray(saved['mean'])))),float(np.max(np.abs(q-np.asarray(saved['quantiles'])))))
                if delta>2e-6:raise ValueError('Frozen gate reference reload mismatch')
                predictions['reference'].append(q)
                _,base=model.predict(transformed,observed=masks,elapsed=np.zeros_like(transformed),source_valid=np.zeros_like(valid))
                predictions['base_only'].append(base)
                # Change only the in-memory gate; immutable trained weights stay intact.
                with torch.no_grad():model.model.gate.weight.zero_();model.model.gate.bias.zero_()
                _,uniform=model.predict(transformed,observed=masks,elapsed=np.zeros_like(transformed),source_valid=valid)
                predictions['uniform_valid_gate'].append(uniform)
                checks.append({'family':family,'outer_year':year,'seed':seed,'receipt_id':digest(receipt),'reference_max_abs_delta':delta})
            for name,values in predictions.items():
                score=quantile_loss(np.asarray(saved['y']),np.mean(values,axis=0),QUANTILES,row['scale'])
                losses[family][name].update({d:float(v) for d,v in zip(saved['decision_time'],score)})
table=[]
for family,values in losses.items():
    dates=sorted(values['reference']);indices,metadata=calendar_indices(dates,8,2000)
    reference=np.asarray([values['reference'][d] for d in dates])
    for case in protocol['cases']:
        if dates!=sorted(values[case]):raise ValueError('Frozen intervention grid mismatch')
        changed=np.asarray([values[case][d] for d in dates])
        table.append({'family':family,'case':case,'n_dates':len(dates),'reference_pinball':float(reference.mean()),
            'intervention_pinball':float(changed.mean()),'reference_minus_intervention':paired_statistics(reference,changed,indices,metadata)})
result={'state':'SUCCEEDED_FROZEN_GATE_DIAGNOSTIC','protocol':protocol,'checks':checks,'comparison_table':table,
    'date_losses':losses,'actual_CUDA_used':True,'new_fits':0,'qualified_for_final':False,'reserved_access':False,'at':now()}
identity=digest(result);commit_bundle(runtime/'artifacts/frozen_gate_controls'/identity,{'results.json':result},{'protocol_id':digest(protocol)})
atomic_json(repo/'reports/frozen_gate_controls.json',result);print(json.dumps(table,indent=2))
import os,sys
# Replace this completed process so its CUDA context is destroyed before the
# next experiment checks ownership. The workflow's stage timeout still applies.
sys.stdout.flush();sys.stderr.flush()
os.execv(sys.executable,[sys.executable,'scripts/run_fixed_gate_ablations.py'])
