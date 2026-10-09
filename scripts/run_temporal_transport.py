"""Frozen 2022 recipe: 2021-trained zero-shot vs 2022-trained refit on 2023."""
import json,sys
from contextlib import nullcontext
import numpy as np
import pandas as pd
from signalforge.runtime import paths,validate_bundle,atomic_json,digest,now
from signalforge.data import auxiliary_rows
from signalforge.inference import infer_bundle
from signalforge.metrics import quantile_loss
from signalforge.models import QUANTILES
from signalforge.resources import research_gpu_stage
from signalforge.statistics import calendar_indices,paired_statistics

repo,runtime=paths();all_families='--all' in sys.argv
prefix='auxiliary_all_family' if all_families else 'auxiliary_cpu'
post=json.loads((repo/'reports'/(prefix+'_postprocessing.json')).read_text())
snapshots=[]
for directory in (runtime/'artifacts/auxiliary_runs').iterdir():
    if not (directory/'receipt.json').exists():continue
    receipt=validate_bundle(directory);doc=json.loads((directory/'results.json').read_text())
    if doc.get('state')=='SUCCEEDED_DIAGNOSTIC' and doc['protocol'].get('compute_scope')==('all' if all_families else 'cpu_only'):
        ids={r['data_id'] for r in doc['results'] if r['state']=='SUCCEEDED'}
        if post['protocol']['canonical_data_id'] in ids or receipt['metadata']['data_id']==json.loads((runtime/'artifacts/model_input_snapshots'/post['protocol']['canonical_data_id']/'receipt.json').read_text())['metadata']['legacy_data_id']:
            snapshots.append((directory,receipt,doc))
if not snapshots:raise RuntimeError('BLOCKED_DATA: corrected immutable temporal reference absent')
directory,receipt,development=sorted(snapshots,key=lambda value:str(value[0]))[0]
reference={(r['family'],r['seed']):r for r in development['results'] if r.get('outer_year')==2022 and r.get('state')=='SUCCEEDED'}
capacity=repo/'reports/auxiliary_capacity_results.json'
if all_families and capacity.exists():
    cap=json.loads(capacity.read_text())
    if cap.get('state')=='SUCCEEDED_DIAGNOSTIC':reference.update({(r['family'],r['seed']):r for r in cap['results'] if r['outer_year']==2022})
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text());origins=auxiliary_rows(acquisition['records']);contexts={}
for i in range(25,len(origins)):
    row=origins.iloc[i]
    if '2023-01-01'<=row.decision_time<'2024-01-01':contexts[pd.Timestamp(row.decision_time).isoformat()]=np.asarray(origins.iloc[i-25:i+1].x.tolist())
testing=origins[(origins.decision_time>='2023-01-01')&(origins.decision_time<'2024-01-01')].copy()
raw=np.asarray([contexts[pd.Timestamp(d).isoformat()] for d in testing.decision_time])
mature=np.isfinite(testing.y.to_numpy())&pd.to_datetime(testing.label_available_at,utc=True).le(pd.Timestamp('2023-12-31T23:59:59.999999999Z')).to_numpy()
dates=testing.loc[mature,'decision_time'].tolist();indices,metadata=calendar_indices(dates,8,2000)
protocol={'experiment':'E06','intervention':'temporal_zero_shot_vs_refit','partition':'2023_diagnostic','seeds':[11,37,71],
    'recipe':'same 2022 earlier-inner-only hyperparameters in both arms','zero_shot_training_cutoff':'2021-12-31',
    'refit_training_cutoff':'2022-12-31','scoring_scale':'common 2022 outer train-only target scale; no 2023 targets fit either scale',
    'forecast_dates':len(testing),'mature_target_dates':int(mature.sum()),'source_tier':'B','source_snapshot_receipt_id':digest(receipt),
    'asset_group_transport':'BLOCKED_DATA: only one qualified physical target entity; Main/Nested source/price gates absent',
    'prospective':False,'reserved_access':False,'promotion_eligible':False}
atomic_json(repo/'reports'/(prefix+'_temporal_transport_protocol.json'),protocol)
table=[];checks=[]
with research_gpu_stage(repo,runtime,'all_family_temporal_transport',1200) if all_families else nullcontext():
    for candidate in post['candidates']:
        family=candidate['candidate_id'].removeprefix('aux_cpu_').removeprefix('aux_all_');old=[];refit=[];scale=None;normalizer=None
        for seed in protocol['seeds']:
            source=reference[(family,seed)];model=runtime/'artifacts/auxiliary'/source['run_id'];validate_bundle(model)
            member=next(m for m in candidate['members'] if m['member_id']==str(seed));new=runtime/member['relative_bundle'];new_receipt=validate_bundle(new)
            if digest(new_receipt)!=member['receipt_id']:raise ValueError('Refit member receipt mismatch')
            new_metric=json.loads((new/'metrics.json').read_text())
            if source['trial']!=new_metric['trial']:raise ValueError('Temporal comparison changed hyperparameter recipe')
            if scale is not None and (scale!=source['scale'] or normalizer!=source['normalizer_id']):raise ValueError('Scoring normalizer differs across seeds')
            scale=source['scale'];normalizer=source['normalizer_id']
            old.append(infer_bundle(model,raw));refit.append(infer_bundle(new,raw))
            checks.append({'family':family,'seed':seed,'zero_shot_run_id':source['run_id'],'refit_receipt_id':member['receipt_id'],'scoring_normalizer_id':normalizer})
        old_mean=np.mean([m for m,q in old],axis=0);old_q=np.mean([q for m,q in old],axis=0)
        new_mean=np.mean([m for m,q in refit],axis=0);new_q=np.mean([q for m,q in refit],axis=0)
        y=testing.y.to_numpy()[mature]
        left=quantile_loss(y,old_q[mature],QUANTILES,scale);right=quantile_loss(y,new_q[mature],QUANTILES,scale)
        table.append({'family':family,'forecast_dates':len(testing),'scored_dates':len(dates),'missing_or_immature_targets':len(testing)-len(dates),
            'zero_shot_pinball':float(left.mean()),'refit_pinball':float(right.mean()),
            'zero_shot_normalized_mse':float(np.mean(((y-old_mean[mature])/scale)**2)),
            'refit_normalized_mse':float(np.mean(((y-new_mean[mature])/scale)**2)),
            'paired_zero_shot_minus_refit':paired_statistics(left,right,indices,metadata)})
result={'state':'SUCCEEDED_FROZEN_TEMPORAL_DIAGNOSTIC','protocol':protocol,'comparison_table':table,'checks':checks,
    'new_model_fits':0,'qualified_for_final':False,'final_access':False,'at':now()}
atomic_json(repo/'reports'/(prefix+'_temporal_transport.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in {'checks','protocol'}}))
