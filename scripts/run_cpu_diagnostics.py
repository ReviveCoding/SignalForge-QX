"""Frozen CPU release robustness and raw-source error traces; never refits/selects."""
import json,sys
from contextlib import nullcontext
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,validate_bundle,digest,now,gpu_lease,file_hash,ensure_gpu_owner
from signalforge.data import auxiliary_rows,eia_events
from signalforge.auxiliary import sequences
from signalforge.diagnostics import rebuild_physical_features
from signalforge.experiment_registry import ExperimentSpec,apply_release_outage
from signalforge.ensemble import FrozenEnsemble
from signalforge.inference import infer_bundle
from signalforge.metrics import quantile_loss
from signalforge.models import QUANTILES
from signalforge.statistics import calendar_indices,paired_statistics
from signalforge.resources import research_gpu_stage

repo,runtime=paths()
all_families='--all' in sys.argv
prefix='auxiliary_all_family' if all_families else 'auxiliary_cpu'
current=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
if current.get('state')!='SUCCEEDED_DIAGNOSTIC' or not current.get('source_integrity_valid',True):
    raise RuntimeError('BLOCKED_DATA: corrected completed development results required')
current_ids={r['data_id'] for r in current['results'] if r.get('state')=='SUCCEEDED'}
if len(current_ids)!=1:raise ValueError('Unique current corpus required')
snapshots=[]
for directory in (runtime/'artifacts/auxiliary_runs').iterdir():
    if not (directory/'receipt.json').exists():continue
    receipt=validate_bundle(directory);doc=json.loads((directory/'results.json').read_text())
    if (doc['protocol'].get('compute_scope')==('all' if all_families else 'cpu_only') and doc.get('state')=='SUCCEEDED_DIAGNOSTIC'
        and {r['data_id'] for r in doc['results'] if r.get('state')=='SUCCEEDED'}==current_ids):
        snapshots.append((directory,receipt,doc))
if not snapshots:raise RuntimeError('BLOCKED_DATA: completed immutable CPU development snapshot required')
directory,receipt,document=sorted(snapshots,key=lambda value:str(value[0]))[0]
config=document['protocol'];families=list(config['families']);seeds=config['seeds']
rows={(r['outer_year'],r['family'],r['seed']):r for r in document['results'] if r['state']=='SUCCEEDED'}
capacity_evidence=None
if all_families and (repo/'reports/auxiliary_capacity_results.json').exists():
    cap_path=repo/'reports/auxiliary_capacity_results.json';cap=json.loads(cap_path.read_text())
    if cap.get('state')=='SUCCEEDED_DIAGNOSTIC':
        if cap['protocol']['data_id'] not in current_ids or cap['protocol']['outer_years']!=config['outer_years'] or cap['protocol']['seeds']!=seeds:
            raise ValueError('Capacity outage control corpus/folds/seeds mismatch')
        families+=cap['protocol']['families']
        rows.update({(r['outer_year'],r['family'],r['seed']):r for r in cap['results'] if r['state']=='SUCCEEDED'})
        capacity_evidence={'path':'reports/auxiliary_capacity_results.json','sha256':file_hash(cap_path),'model_bundles':len(cap['results'])}
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text());original=auxiliary_rows(acquisition['records'])
base_frame,base_x=sequences(original);events=eia_events(acquisition['records'])
original_index={pd.Timestamp(date).isoformat():i for i,date in enumerate(base_frame.decision_time)}
y_by_date={pd.Timestamp(r.decision_time).isoformat():r for r in base_frame.itertuples()}
protocol={'created_at':now(),'experiment':'E05','intervention':'frozen_input','retrained':False,
          'source_snapshot_receipt':digest(receipt),'families':families,'seeds':seeds,
          'schedules':{'seven_day_delay':[{'source':'eia','kind':'delay','days':7,'start':'2011-01-01T00:00Z','end':'2024-01-01T00:00Z'}],
            'january_four_week_blackout':[{'source':'eia','kind':'blackout','start':f'{year}-01-01T00:00Z',
               'end':(pd.Timestamp(f'{year}-01-01T00:00Z')+pd.Timedelta(days=28)).isoformat()} for year in config['outer_years']]},
          'labels_and_normalizers':'held fixed from original development grid','fallback_policy':'carry genuinely known older complete releases; recompute release/reference ages',
          'claim_boundary':'Exploratory physical-only Tier B robustness; no retrained-ablation, alpha or causal attribution claim','final_access':False}
protocol['experiment_id']=ExperimentSpec('E05','Auxiliary','development','frozen_input',False,protocol['schedules']).validate()
protocol['capacity_controls']=capacity_evidence
atomic_json(repo/'reports'/(prefix+'_robustness_protocol.json'),protocol)
dependencies={'physical_stocks':['eia'],'release_age':['eia'],'reference_age':['eia'],
              'context':['physical_stocks','release_age','reference_age']}
scenarios={};traces={}
for scenario,schedule in protocol['schedules'].items():
    perturbed,invalid,trace=apply_release_outage(events,schedule,dependencies)
    rebuilt=rebuild_physical_features(original,perturbed);frame,x=sequences(rebuilt)
    if frame.decision_time.tolist()!=base_frame.decision_time.tolist() or not np.array_equal(frame.y,base_frame.y):
        raise ValueError('Intervention changed outcome-blind grid or labels')
    scenarios[scenario]=x;traces[scenario]={'invalidated_descendants_rebuilt':invalid,'release_intervention':trace,
                  'changed_contexts':int(np.any(np.abs(x-base_x)>1e-12,axis=(1,2)).sum())}
losses={f:{} for f in families};changed={name:{f:{} for f in families} for name in scenarios};forecast_rows={f:{} for f in families}
lease=research_gpu_stage(repo,runtime,'all_family_frozen_release_outages',1800) if all_families else nullcontext()
with lease:
 for year in config['outer_years']:
    for family in families:
        if all_families:ensure_gpu_owner()
        members={};perturbed_members={name:{} for name in scenarios};model_ids=[]
        for seed in seeds:
            row=rows[(year,family,seed)];model_ids.append(row['run_id'])
            model=runtime/'artifacts/auxiliary'/row['run_id'];validate_bundle(model)
            saved=json.loads((model/'predictions.json').read_text());dates=[pd.Timestamp(d).isoformat() for d in saved['decision_time']];y=np.array(saved['y']);scale=row['scale']
            indices=[original_index[pd.Timestamp(date).isoformat()] for date in dates]
            if not np.array_equal(y,base_frame.iloc[indices].y.to_numpy()):raise ValueError('Stored outcomes differ from corrected grid')
            members[str(seed)]=(np.array(saved['mean']),np.array(saved['quantiles']))
            for name,x in scenarios.items():perturbed_members[name][str(seed)]=infer_bundle(model,x[indices])
        ensemble=FrozenEnsemble([str(seed) for seed in seeds]);mean,q=ensemble.predict(members)
        for date,loss,mu,quantiles in zip(dates,quantile_loss(y,q,QUANTILES,scale),mean,q):
            losses[family][date]=float(loss);forecast_rows[family][date]={'mean':float(mu),'quantiles':quantiles.tolist(),
                                   'scale':scale,'model_ids':model_ids,'normalizer_id':row['normalizer_id']}
        for name,members_ in perturbed_members.items():
            mu,q=ensemble.predict(members_)
            for date,loss in zip(dates,quantile_loss(y,q,QUANTILES,scale)):changed[name][family][date]=float(loss)
dates=sorted(losses['ridge']);indices,metadata=calendar_indices(dates,8,2000)
table=[]
for name in scenarios:
    for family in families:
        if sorted(changed[name][family])!=dates:raise ValueError('Robustness paired grid mismatch')
        old=np.array([losses[family][date] for date in dates]);new=np.array([changed[name][family][date] for date in dates])
        table.append({'scenario':name,'family':family,'n_market_dates':len(dates),'original_pinball':float(old.mean()),
          'perturbed_pinball':float(new.mean()),'pinball_increase':float((new-old).mean()),
          'paired':paired_statistics(old,new,indices,metadata),'feature_independent_baseline':family in {'historical','ewma'}})
errors=[]
for date in sorted(dates,key=lambda date:losses['ridge'][date],reverse=True)[:5]:
    source=y_by_date[pd.Timestamp(date).isoformat()];forecast=forecast_rows['ridge'][date]
    errors.append({'decision_time':date,'selection':'post-hoc five largest Ridge development losses; never changes the cohort/models',
       'feature_raw_sha256':source.raw_hash,'target_raw_sha256':source.target_raw_hash,
       'feature_available_at':source.max_dependency_available_at,'target_available_at':source.label_available_at,
       'target_issue_date':source.target_issue_date,'target_million_barrels':source.y,'normalized_pinball':losses['ridge'][date],
       **forecast,'controller_state':'BLOCKED_PRICE_P1; inventory target is not a qualified investable cash-return forecast'})
result={'created_at':now(),'state':'SUCCEEDED_FROZEN_ALL_FAMILY_DIAGNOSTIC' if all_families else 'SUCCEEDED_FROZEN_CPU_DIAGNOSTIC','protocol':protocol,'comparison_table':table,
        'source_traces':traces,'error_cases':errors,'n_market_dates':len(dates),'same_draw_indices':metadata['index_id'],
        'no_refitting_or_model_seed_selection':True,'no_exact_causal_PnL_attribution':True,'final_access':False}
atomic_json(repo/'reports'/(prefix+'_robustness.json'),result)
print(json.dumps({'state':result['state'],'n_dates':len(dates),'rows':len(table),'error_cases':len(errors)}))
