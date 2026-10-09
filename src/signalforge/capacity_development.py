"""Measured same-information, capacity-matched Auxiliary CUDA comparisons."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import paths,atomic_json,digest,file_hash,commit_bundle,validate_bundle,gpu_lease,now
from .data import auxiliary_rows
from .auxiliary import sequences,FAMILIES
from .development import preprocess,trial_grid,CPU_FAMILIES
from .panels import track_folds
from .input_identity import canonical_data_id
from .capacity import CapacityMatchedCUDA
from .resources import require_current_resources
from .experiments import ComputeBudget,TrialLedger
from .metrics import quantile_loss
from .models import QUANTILES
from .statistics import calendar_indices,paired_statistics

FAMILIES_MATCHED={'concat_capacity_gru':'rgmf_gru','concat_capacity_transformer':'rgmf_transformer'}


def fit_capacity_artifact(repo,runtime,budget,family,trial,seed,training,testing,arrays,identity,data_id):
    destination=runtime/'artifacts/auxiliary'/identity
    if (destination/'receipt.json').exists():validate_bundle(destination);return json.loads((destination/'metrics.json').read_text())
    require_current_resources(repo,runtime)
    tx,vx,scale,normalizer,transform=arrays;before=time.monotonic()
    with budget.charge(identity,300):
        width,lr=trial['neural']
        model=CapacityMatchedCUDA(family.removeprefix('concat_capacity_'),width=width,lr=lr,epochs=100,seed=seed)
        model.fit(tx,training.y.to_numpy(),scale,checkpoint_path=runtime/'checkpoints/capacity'/(identity+'.pt'))
        mean,q=model.predict(vx)
    metric={'state':'SUCCEEDED','family':family,'seed':seed,'trial':trial,'data_id':data_id,'normalizer_id':normalizer,'scale':scale,
        'run_id':identity,'seconds':time.monotonic()-before,'pinball':float(quantile_loss(testing.y.to_numpy(),q,QUANTILES,scale).mean()),
        'parameter_count':model.parameter_count,'capacity_spec':model.capacity_spec,'evidence_kind':'development','actual_CUDA_used':True}
    predictions={'decision_time':pd.to_datetime(testing.decision_time,utc=True).astype(str).tolist(),'y':testing.y.tolist(),'mean':mean.tolist(),'quantiles':q.tolist()}
    with tempfile.TemporaryDirectory(dir=runtime/'tmp') as temporary:
        path=Path(temporary)/'model.bin';model.save(path)
        commit_bundle(destination,{'model.bin':path.read_bytes(),'metrics.json':metric,'transform.json':transform.manifest(),'predictions.json':predictions},
            {'data_id':data_id,'run_id':identity,'capacity_implementation_hash':file_hash(repo/'src/signalforge/capacity.py'),'reserved_access':False})
    return metric


def run_capacity_development(repo,runtime):
    document=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
    if document.get('state')!='SUCCEEDED_DIAGNOSTIC' or document['protocol']['compute_scope']!='all':raise RuntimeError('BLOCKED_DATA: complete core comparison required')
    a=json.loads((repo/'reports/eia_development_acquisition.json').read_text());frame,x=sequences(auxiliary_rows(a['records']))
    data_id=canonical_data_id(a,frame,x);study=json.loads((repo/'configs/study.json').read_text());folds,blocked=track_folds(frame,study,'Auxiliary-C')
    if blocked:raise RuntimeError('BLOCKED_DATA: full matched development folds unavailable')
    reference={(r['outer_year'],r['family'],r['seed']):r for r in document['results'] if r.get('state')=='SUCCEEDED'}
    profile=json.loads((repo/'configs/local_rtx4090_laptop.json').read_text())
    protocol={'families':list(FAMILIES_MATCHED),'target_families':FAMILIES_MATCHED,'outer_years':sorted(folds),'seeds':study['hpo']['development_seeds'],
        'epochs':100,'inner_trials':20,'target_capacity':'reference RGMF width selected only on the earlier inner fold of that year',
        'capacity_tolerance':.01,'tie_policy':'closest sqrt(2)*reference width among controls within 1%; then parameter gap/width',
        'learning_rates':np.geomspace(.0003,.01,20).tolist(),'information':'identical full transformed 26-week input vector',
        'data_id':data_id,'promotion_eligible':False,'claim_scope':'Reconstructed Tier B diagnostic capacity control; not strict final qualification',
        'pilot_new_fits':12,'pilot_total_completed_ceiling':profile['budget']['pilot_completed_fit_ceiling'],'reserved_access':False}
    method=digest({name:file_hash(repo/'src/signalforge'/(name+'.py')) for name in ['capacity','capacity_development','models','neural','features']})
    protocol['method_id']=method;atomic_json(repo/'reports/auxiliary_capacity_protocol.json',protocol)
    budget=ComputeBudget(runtime/'ledger/development_compute.sqlite',profile['budget']['development_gpu_seconds'])
    ledger=TrialLedger(runtime/'ledger/capacity_trials.sqlite');pilots=[];results=[]
    proof=json.loads((repo/'reports/canonical_cache_upgrade_qualification.json').read_text())
    proof_directory=runtime/'artifacts/source_correction_cache_proofs'/proof['proof_id'];validate_bundle(proof_directory)
    parents=json.loads((proof_directory/'proof.json').read_text())['parents'];old_pilots=[]
    for family in FAMILIES:
        for year in [2018,2019]:
            for seed in [11,37,71]:
                key=digest({'partition':'inner_validation','year':year,'family':family,'seed':seed,'trial':trial_grid(family)[-1]})
                directory=runtime/'artifacts/auxiliary'/parents[key];validate_bundle(directory)
                metric=json.loads((directory/'metrics.json').read_text());old_pilots.append(metric)
    old_pilot_seconds=sum(r['seconds'] for r in old_pilots if r['family'] not in CPU_FAMILIES)
    if len(old_pilots)+12>profile['budget']['pilot_completed_fit_ceiling']:raise RuntimeError('PAUSED_BUDGET: cumulative pilot fit ceiling')
    def fit(family,year,stage,trial,seed):
        fold=folds[year];train,test,cut=(fold['inner'],fold['valid'],fold['inner_cutoff']) if stage=='inner' or stage=='pilot' else (fold['train'],fold['test'],fold['outer_cutoff'])
        arrays=preprocess(x,train,test,cut)
        config={'partition':'inner_validation' if stage!='outer' else 'outer_development','family':family,'year':year,'trial':trial,'seed':seed,
            'method_id':method,'data_id':data_id,'normalizer_id':arrays[3],'stage':stage}
        identity=digest(config)
        trial_id=ledger.register({**config,'partition':'inner_validation'}) if stage!='outer' else None
        try:
            row=fit_capacity_artifact(repo,runtime,budget,family,trial,seed,train,test,arrays,identity,data_id)
            if trial_id and ledger.get(trial_id)[2]=='PLANNED':ledger.finish(trial_id,'SUCCEEDED',row['pinball'])
            return {**row,'outer_year':year,'stage':stage}
        except Exception as error:
            if trial_id and ledger.get(trial_id)[2]=='PLANNED':ledger.finish(trial_id,'PAUSED_BUDGET' if 'PAUSED_BUDGET' in str(error) else 'FAILED_PERMANENT',failure=str(error))
            if str(error).startswith(('PAUSED_BUDGET','BLOCKED_RESOURCE','BLOCKED_GPU')):raise
            return {'state':'FAILED_PERMANENT','family':family,'outer_year':year,'stage':stage,'seed':seed,'run_id':identity,'failure':str(error)}
    try:
        with gpu_lease(runtime):
            for family in FAMILIES_MATCHED:
                for year in [2018,2019]:
                    for seed in protocol['seeds']:
                        if old_pilot_seconds+sum(r.get('seconds',0) for r in pilots)+300>profile['budget']['pilot_gpu_seconds']:
                            raise RuntimeError('PAUSED_BUDGET: cumulative pilot wall-time ceiling')
                        pilots.append(fit(family,year,'pilot',{'regularization':1.,'neural':[64,.01]},seed))
            if any(r['state']!='SUCCEEDED' for r in pilots):raise RuntimeError('BLOCKED_RESOURCE: capacity pilot failure; no full capacity run')
            estimate=2*sum(max(r['seconds'] for r in pilots if r['family']==f)*115 for f in FAMILIES_MATCHED)
            bill={'state':'READY' if estimate<=budget.remaining else 'PAUSED_BUDGET','estimated_gpu_seconds':estimate,'remaining_gpu_seconds':budget.remaining,
                'new_pilot_completed':len(pilots),'cumulative_pilot_completed':len(old_pilots)+len(pilots),
                'cumulative_pilot_seconds':old_pilot_seconds+sum(r['seconds'] for r in pilots),'actual_physical_gpus':1,'reserved_access':False}
            atomic_json(repo/'reports/auxiliary_capacity_bill.json',bill)
            if bill['state']!='READY':raise RuntimeError('PAUSED_BUDGET: measured capacity bill')
            for year in sorted(folds):
                for family,target in FAMILIES_MATCHED.items():
                    ref=reference[(year,target,11)];width=ref['trial']['neural'][0]
                    trials=[fit(family,year,'inner',{'regularization':1.,'neural':[width,lr]},11) for lr in protocol['learning_rates']]
                    successful=[(i,r) for i,r in enumerate(trials) if r['state']=='SUCCEEDED']
                    if not successful:results.append({'state':'FAILED_ALL_INNER','family':family,'outer_year':year,'inner_trials':trials});continue
                    index,winner=min(successful,key=lambda item:(item[1]['pinball'],item[0]))
                    for seed in protocol['seeds']:
                        row=fit(family,year,'outer',winner['trial'],seed)
                        results.append({**row,'inner_selection_id':winner['run_id'],'inner_trial_index':index,'reference_run_id':reference[(year,target,seed)]['run_id'],'inner_attempts':trials})
                    atomic_json(repo/'reports/auxiliary_capacity_results.json',{'state':'RUNNING','protocol':protocol,'results':results,'pilots':pilots,'qualified_for_final':False})
    finally:budget.close();ledger.close()
    result={'state':'SUCCEEDED_DIAGNOSTIC' if len(results)==30 and all(r['state']=='SUCCEEDED' for r in results) else 'PARTIAL','protocol':protocol,'results':results,'pilots':pilots,'qualified_for_final':False}
    atomic_json(repo/'reports/auxiliary_capacity_results.json',result)
    commit_bundle(runtime/'artifacts/capacity_runs'/digest({'protocol':protocol,'runs':[r.get('run_id') for r in results]}),{'results.json':result},{'data_id':data_id,'reserved_access':False})
    return result
