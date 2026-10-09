"""Real Auxiliary development null/leakage diagnostics; all invalid/nonpromotable."""
import json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from signalforge.runtime import paths,validate_bundle,atomic_json,commit_bundle,digest,now
from signalforge.data import auxiliary_rows,eia_events
from signalforge.auxiliary import sequences
from signalforge.development import preprocess
from signalforge.features import purged_training
from signalforge.models import Statistical,QUANTILES
from signalforge.metrics import quantile_loss
from signalforge.experiment_registry import invalid_control,ExperimentSpec
from signalforge.diagnostics import rebuild_physical_features
from signalforge.statistics import calendar_indices,paired_statistics
from signalforge.input_identity import canonical_data_id

repo,runtime=paths();acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
original=auxiliary_rows(acquisition['records']);frame,x=sequences(original)
if pd.to_datetime(frame.decision_time,utc=True).ge('2024-01-01T00:00Z').any():raise PermissionError('Development-only controls')
corpus=validate_bundle(runtime/'artifacts/model_input_snapshots'/canonical_data_id(acquisition,frame,x))
parents=[]
for directory in (runtime/'artifacts/auxiliary_runs').iterdir():
    if not (directory/'receipt.json').exists():continue
    receipt=validate_bundle(directory);document=json.loads((directory/'results.json').read_text())
    if document.get('state')=='SUCCEEDED_DIAGNOSTIC' and document['protocol'].get('compute_scope')=='cpu_only' and receipt['metadata']['data_id']==corpus['metadata']['legacy_data_id']:
        parents.append((directory,receipt,document))
if not parents:raise RuntimeError('BLOCKED_DATA: corrected CPU reference snapshot absent')
parent,receipt,document=sorted(parents,key=lambda value:str(value[0]))[0]
recipes={(r['outer_year'],r['seed']):r for r in document['results'] if r.get('family')=='ridge' and r.get('state')=='SUCCEEDED'}
wrong=invalid_control(eia_events(acquisition['records']),'wrong_release','development')
wrong_frame,wrong_x=sequences(rebuild_physical_features(original,wrong))
assert wrong_frame.decision_time.tolist()==frame.decision_time.tolist()
# Sentinel intentionally exposes the future target. It is an invalid positive
# control, never an available feature or a source of scientific model selection.
sentinel=np.concatenate([x,np.zeros((*x.shape[:2],1))],axis=-1);sentinel[:,-1,-1]=frame.y.to_numpy()
config={'experiment':'E04','track':'Auxiliary-C','partition':'development','family':'ridge','seeds':[11,37,71],
        'controls':['train_target_permutation','wrong_release_clock','future_target_sentinel'],'promotion_eligible':False,
        'parameter_source':'the same earlier inner-only CPU Ridge recipe for each outer fold','final_access':False,
        'reference_snapshot_receipt_id':digest(receipt),'source_tier':'B','invalid_clock_and_sentinel_tier':'C',
        'interpretation':'Real reconstructed-source negative/invalid positive diagnostics; not strong CUDA-baseline research qualification',
        'latest_revision':'BLOCKED_ORIGINAL_VINTAGE: no authenticated revision history; no fabricated latest-vintage experiment'}
config['experiment_id']=ExperimentSpec('E04','Auxiliary-C','development','invalid_control',True,config,False).validate()
atomic_json(repo/'reports/auxiliary_invalid_cpu_controls_protocol.json',config)
losses={name:{} for name in ['reference',*config['controls']]};checks=[]
for year in document['protocol']['outer_years']:
    testing=frame[(frame.decision_time>=f'{year}-01-01')&(frame.decision_time<f'{year+1}-01-01')]
    cutoff=pd.Timestamp(f'{year}-01-01T00:00Z')-pd.Timedelta(nanoseconds=1)
    training=purged_training(frame,cutoff,list(zip(testing.label_start,testing.label_end)))
    member_predictions={name:[] for name in losses}
    scale=None
    for seed in config['seeds']:
        reference=recipes[(year,seed)];directory=runtime/'artifacts/auxiliary'/reference['run_id'];validate_bundle(directory)
        stored=json.loads((directory/'predictions.json').read_text());assert pd.to_datetime(stored['decision_time'],utc=True).tolist()==pd.to_datetime(testing.decision_time,utc=True).tolist()
        member_predictions['reference'].append(np.asarray(stored['quantiles']))
        for name,raw in [('train_target_permutation',x),('wrong_release_clock',wrong_x),('future_target_sentinel',sentinel)]:
            tx,vx,scale,normalizer,transform=preprocess(raw,training,testing,cutoff)
            y=training.y.to_numpy().copy()
            if name=='train_target_permutation':y=y[np.random.default_rng(seed+year).permutation(len(y))]
            identity=digest({'protocol':config,'control':name,'year':year,'seed':seed,'transform':transform.manifest(),
                'tx':tx.tolist(),'vx':vx.tolist(),'training_y':y.tolist(),'recipe':reference['trial'],'normalizer_id':normalizer})
            destination=runtime/'artifacts/auxiliary_invalid_controls'/identity
            if (destination/'receipt.json').exists():
                validate_bundle(destination);q=np.asarray(json.loads((destination/'predictions.json').read_text())['quantiles'])
            else:
                model=Statistical('ridge',reference['trial']['regularization']).fit(tx.reshape(len(tx),-1),y)
                mean,q=model.predict(vx.reshape(len(vx),-1))
                with tempfile.TemporaryDirectory(dir=runtime/'tmp') as temporary:
                    path=Path(temporary)/'model.bin';model.save(path)
                    commit_bundle(destination,{'model.bin':path.read_bytes(),'transform.json':transform.manifest(),
                        'predictions.json':{'decision_time':testing.decision_time.tolist(),'mean':mean.tolist(),'quantiles':q.tolist()}},
                        {'promotion_eligible':False,'experiment':'E04','control':name,'normalizer_id':normalizer,'reserved_access':False})
            member_predictions[name].append(q);checks.append({'control':name,'outer_year':year,'seed':seed,'artifact_id':identity})
        assert scale==reference['scale'] and normalizer==reference['normalizer_id']
    for name,predictions in member_predictions.items():
        for date,loss in zip(testing.decision_time,quantile_loss(testing.y.to_numpy(),np.mean(predictions,axis=0),QUANTILES,scale)):losses[name][date]=float(loss)
dates=sorted(losses['reference']);indices,metadata=calendar_indices(dates,8,2000)
table=[]
for name in config['controls']:
    reference=np.asarray([losses['reference'][d] for d in dates]);control=np.asarray([losses[name][d] for d in dates])
    table.append({'control':name,'reference_pinball':float(reference.mean()),'control_pinball':float(control.mean()),
        'paired_reference_minus_control':paired_statistics(reference,control,indices,metadata),'promotion_eligible':False,'n_market_dates':len(dates)})
result={'state':'SUCCEEDED_NONPROMOTABLE_CPU_CONTROLS','protocol':config,'comparison_table':table,'checks':checks,
        'n_actual_control_bundles':len(checks),'n_market_dates':len(dates),'actual_CUDA_used':False,'final_access':False}
atomic_json(repo/'reports/auxiliary_invalid_cpu_controls.json',result);print(json.dumps({k:v for k,v in result.items() if k not in {'checks','protocol'}}))
