"""All-row paired BAR evaluation and independent verification; no retuning."""
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from signalforge.runtime import atomic_json,commit_bundle,validate_bundle,file_hash,digest,now,code_hash
from signalforge.metrics import quantile_loss
from signalforge.v33_comparison import paired_comparison,COLS,Q
from policy_v3.v33_budget_free import ObservationMeter,device_lease,policy_source_hash
from bar_model import BAR,native_predict


def frame(meta,z,q,seed):
    f=pd.DataFrame(meta['date_asset_keys'],columns=['decision_time','asset']);f['seed']=seed;f['target']=z['target'];f['scale']=z['scale'];f[COLS]=q;return f

def diagnostics(f):
    q=f[COLS].to_numpy();y=f.target.to_numpy();scale=f.scale.to_numpy();error=y[:,None]-q;perq=np.maximum(Q*error,(Q-1)*error);norm=perq/scale[:,None];loss=norm.mean(1);absolute=perq.mean(1)
    table=f[['decision_time','asset','seed']].copy();table['normalized_pinball']=loss;table['raw_pinball']=absolute
    perseed=table.groupby('seed')[['normalized_pinball','raw_pinball']].mean().to_dict('index');seed_values=[x['normalized_pinball'] for x in perseed.values()]
    return {'normalized_pinball':float(table.groupby('decision_time').normalized_pinball.mean().mean()),'raw_pinball':float(absolute.mean()),'per_quantile_normalized_pinball':norm.mean(0).tolist(),'per_quantile_raw_pinball':perq.mean(0).tolist(),'empirical_coverage':(y[:,None]<=q).mean(0).tolist(),'crossing_count':int((np.diff(q,axis=1)<0).sum()),'max_absolute_forecast':float(np.abs(q).max()),'p99_absolute_forecast':float(np.quantile(np.abs(q).max(1),.99)),'forecast_rows_abs_over100':int((np.abs(q).max(1)>100).sum()),'mean_90percent_interval_raw':float((q[:,4]-q[:,0]).mean()),'mean_80percent_interval_raw':float((q[:,3]-q[:,1]).mean()),'mean_90percent_interval_normalized':float(((q[:,4]-q[:,0])/scale).mean()),'per_seed':perseed,'seed_loss_range':float(np.ptp(seed_values)),'seed_loss_std_descriptive':float(np.std(seed_values)),'per_asset':table.groupby('asset')[['normalized_pinball','raw_pinball']].mean().to_dict('index'),'distinct_dates':int(f.decision_time.nunique()),'retained_prediction_rows':len(f)}

def date_loss(f):
    values=quantile_loss(f.target.to_numpy(),f[COLS].to_numpy(),Q,f.scale.to_numpy());g=f[['decision_time','seed']].copy();g['loss']=values
    return g.groupby(['decision_time','seed']).loss.mean().groupby('decision_time').mean().sort_index()

def uncertainty(old,new,config):
    a,b=date_loss(old),date_loss(new);assert a.index.equals(b.index);d=a.to_numpy()-b.to_numpy();result={};rng=np.random.default_rng(config['seed'])
    for block in config['date_blocks']:
        count=int(np.ceil(len(d)/block));starts=rng.integers(0,len(d)-block+1,size=(config['draws'],count));indices=(starts[:,:,None]+np.arange(block)).reshape(config['draws'],-1)[:,:len(d)];means=d[indices].mean(1)
        result[str(block)]={'date_equal_mean_loss_reduction':float(d.mean()),'descriptive_95percent_block_interval':np.quantile(means,[.025,.975]).tolist(),'independent_dates':len(d),'draws':config['draws'],'not_confirmatory':True}
    return result

def analyse(R,T,c,plan,external_hash,intact,progress):
    fits=json.loads((R/'reports/bar_fits_v1.json').read_text());assert fits['completed_candidates']==6;results={};decisions=[]
    v2=json.loads((R/'configs/v33_minimal_execution_v2r1.json').read_text());v2fits=json.loads((R/'reports/v33_minimal_study_v2.json').read_text())['results']
    for track,year in [('Main-A',2020),('Nested-B',2022)]:
        key=track+':'+str(year);bundle=T/plan['data_bundles'][key];validate_bundle(bundle);z=np.load(bundle/'arrays.npz');meta=json.loads((bundle/'metadata.json').read_text());oldbundle=T/v2['data_bundles'][key];validate_bundle(oldbundle);oz=np.load(oldbundle/'arrays.npz');assert np.array_equal(oz['target'],z['target']) and np.array_equal(oz['scale'],z['scale'])
        rows=[];controls={'original_RGMF':[],'SIA_v2':[],'strong_I0_baseline':[],'fixed_BAR_anchor':[]};fallback_rows=0
        for seed in [11,37,71]:
            fit=next(r for r in fits['results'] if (r['track'],r['year'],r['seed'])==(track,year,seed));b=T/fit['relative_bundle'];validate_bundle(b);prediction=np.load(b/'predictions.npz');rows.append(frame(meta,z,prediction['quantiles'],seed));fallback_rows+=int(prediction['fallback'].sum())
            controls['original_RGMF'].append(frame(meta,z,oz['old_rgmf_'+str(seed)],seed));controls['strong_I0_baseline'].append(frame(meta,z,oz['strong_baseline_'+str(seed)],seed));controls['fixed_BAR_anchor'].append(frame(meta,z,z['test_base'],seed))
            sr=next(r for r in v2fits if (r['track'],r['year'],r['seed'])==(track,year,seed));sb=T/sr['relative_bundle'];validate_bundle(sb);controls['SIA_v2'].append(frame(meta,z,np.load(sb/'predictions.npz')['quantiles'],seed))
        bar=pd.concat(rows,ignore_index=True);metadata={'fold_id':meta['fold_id'],'source_id':meta['source_id'],'normalizer_id':meta['normalizer_id'],'target_contract_id':meta['target_id']};out={'BAR':diagnostics(bar),'fallback_rows':fallback_rows,'source_ood_rows_per_seed':meta['source_ood_test_rows'],'comparisons':{}}
        for name,frames in controls.items():
            control=pd.concat(frames,ignore_index=True);comparison=paired_comparison(control,bar,metadata,metadata);comparison['control_diagnostics']=diagnostics(control);comparison['normalized_loss_difference_BAR_minus_control']=comparison['new']['normalized_pinball']-comparison['old']['normalized_pinball'];comparison['percent_improvement']=100*comparison['relative_pinball_gain'];comparison['block_uncertainty']=uncertainty(control,bar,c['uncertainty']);out['comparisons'][name]=comparison
        strong=out['comparisons']['strong_I0_baseline'];anchor=out['comparisons']['fixed_BAR_anchor'];newcoverage=np.array(out['BAR']['empirical_coverage']);oldcoverage=np.array(strong['control_diagnostics']['empirical_coverage']);newerr=np.abs(newcoverage-Q);olderr=np.abs(oldcoverage-Q)
        conditions={'strong_baseline_gain_at_least_1percent':strong['relative_pinball_gain']>=.01,'own_anchor_gain_at_least_1percent':anchor['relative_pinball_gain']>=.01,'mean_coverage_error_deterioration_at_most_2pp':float(newerr.mean()-olderr.mean())<=.02,'each_tail_error_deterioration_at_most_5pp':bool((newerr[[0,1,3,4]]-olderr[[0,1,3,4]]<=.05).all()),'stable':out['BAR']['crossing_count']==out['BAR']['forecast_rows_abs_over100']==0}
        out['registered_conditions']=conditions;out['registered_conditions_passed']=all(conditions.values());decisions.append(out['registered_conditions_passed']);results[key]=out
    receipt={'state':'COMPLETED_BAR_POST_RESULT_EXPLORATORY','protocol_id':c['protocol_id'],'configuration_id':c['configuration_id'],'plan_id':plan['plan_id'],'external_study_hash':external_hash(),'policy_source_hash':policy_source_hash(),'frozen_v2_hash':code_hash(R),'results':results,'decision':'PROMISING_RETROSPECTIVE_REQUIRES_SEPARATE_REPLICATION' if all(decisions) else 'NEGATIVE_OR_INCONCLUSIVE_STOP_EXPANSION','promising_registered_rule_met_both_tracks':all(decisions),'statistical_claim':'no confirmatory/significance claim; 52 dates per track, seeds not independent samples; post-result exploratory','calibration_fit_executed':False,'old_BAR_gate_not_changed':True,'reserved_access':False,'qualified_for_final':False,'preservation':intact()}
    atomic_json(R/'reports/bar_exploratory_metrics_v1.json',receipt);progress('BAR_PAIRED_EVALUATION_COMPLETE',decision=receipt['decision'],summary={k:{'BAR':v['BAR']['normalized_pinball'],'strong':v['comparisons']['strong_I0_baseline']['old']['normalized_pinball'],'strong_gain_percent':v['comparisons']['strong_I0_baseline']['percent_improvement']} for k,v in results.items()})
    return receipt

def independently_validate(R,T,c,plan,external_hash,intact,progress):
    from policy_v3.v33_budget_free import admission
    qual=json.loads((R/'reports/bar_execution_qualification_v1.json').read_text());admission(plan,qual,experiment_requested=True);assert plan['external_study_hash']==external_hash() and c['policy_source_hash']==policy_source_hash();fits=json.loads((R/'reports/bar_fits_v1.json').read_text());metrics=json.loads((R/'reports/bar_exploratory_metrics_v1.json').read_text());assert len(fits['results'])==6
    meter=ObservationMeter(T/'ledger/v33_budget_free_v3_bar_validation_v1.sqlite');checks=[]
    try:
        with device_lease(2*1024**3):
            for track,year in [('Main-A',2020),('Nested-B',2022)]:
                key=track+':'+str(year);b=T/plan['data_bundles'][key];validate_bundle(b);z=np.load(b/'arrays.npz');meta=json.loads((b/'metadata.json').read_text());allq=[];states=[]
                for fit in [r for r in fits['results'] if (r['track'],r['year'])==(track,year)]:
                    folder=T/fit['relative_bundle'];validate_bundle(folder);assert json.loads((folder/'fit.json').read_text())==fit;identity=digest({'phase':'independent_reload','fit_id':fit['fit_id'],'external_hash':external_hash()})
                    with meter.attempt(identity,metadata={'fit_id':fit['fit_id'],'purpose':'independent prediction verification, no training'}) as observed:
                        saved=torch.load(folder/'model.pt',map_location='cuda',weights_only=False);model=BAR(**saved['spec']).cuda();model.load_state_dict(saved['state']);q,fb=native_predict(model,z['test_x'],meta['groups'],z['test_base'],z['scale'],z['test_valid'],z['test_ood']);stored=np.load(folder/'predictions.npz');assert np.array_equal(q,stored['quantiles']) and np.array_equal(fb,stored['fallback']);assert np.array_equal(q[fb],z['test_base'][fb]);assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all();observed.observe(peak_vram_bytes=int(torch.cuda.max_memory_allocated()),outcome={'predictions_bitwise_identical':True});del model;torch.cuda.empty_cache()
                    allq.append(q);states.append(fit)
                error=z['target'][None,:,None]-np.stack(allq);loss=np.maximum(Q*error,(Q-1)*error);norm=(loss/z['scale'][None,:,None]).mean();absolute=loss.mean();assert np.isclose(norm,metrics['results'][key]['BAR']['normalized_pinball'],rtol=1e-12) and np.isclose(absolute,metrics['results'][key]['BAR']['raw_pinball'],rtol=1e-12)
                assert len(meta['date_asset_keys'])==52*8 and metrics['results'][key]['BAR']['retained_prediction_rows']==1248
                for comparison in metrics['results'][key]['comparisons'].values():assert comparison['rows_dropped']==0
                assert all(pd.Timestamp(t)<=pd.Timestamp(trace['cutoff']) for trace in meta['OOF_traces'] for t in [trace['max_label_available_at']])
                checks.append({'track':track,'year':year,'candidates':3,'independent_dates':52,'rows':1248,'all_reload_equal':True,'independent_numpy_scores_match':True,'OOF_maturity_traces_valid':True})
        observed=meter.rows()
    finally:meter.close()
    study_meter=ObservationMeter(T/'ledger/v33_budget_free_v3_bar_study_v1.sqlite');integration_meter=ObservationMeter(T/'ledger/v33_budget_free_v3_bar_integration_v1.sqlite')
    try:
        records=study_meter.rows();irecords=integration_meter.rows();assert len([r for r in records if r['state']=='SUCCEEDED'])==6;assert {json.loads(r['metadata'])['fit_id'] for r in records if r['state']=='SUCCEEDED'}=={r['fit_id'] for r in fits['results']};assert any(r['state']=='FAILED' for r in irecords)
    finally:study_meter.close();integration_meter.close()
    original=R.with_name('SignalForge-QX');isolation=json.loads((R/'reports/v33_isolation_receipt.json').read_text(encoding='utf-8-sig'))
    for p,h in isolation['copied_files'].items():assert file_hash(original/p.replace('\\','/'))==h
    v2=R/'reports/v33_minimal_results_v2.json';old=json.loads(v2.read_text());assert old['bar_state']=='DEFERRED_PREDECLARED_SIGNAL_RULE_NOT_MET'
    preservation=intact();receipt={'state':'PASSED_INDEPENDENT_BAR_VALIDATION','external_study_hash':external_hash(),'policy_source_hash':policy_source_hash(),'plan_id':plan['plan_id'],'configuration_id':c['configuration_id'],'frozen_v2_hash':code_hash(R),'checks':checks,'study_meter_records':records,'integration_meter_records':irecords,'independent_reload_meter_records':observed,'observed_study_gpu_wall_seconds':sum(r['elapsed_seconds'] for r in records),'observed_integration_gpu_wall_seconds':sum(r['elapsed_seconds'] for r in irecords),'observed_validation_gpu_wall_seconds':sum(r['elapsed_seconds'] for r in observed),'no_time_caps_or_bill_or_manual_cost_gate':True,'original_357_files_unchanged':True,'preservation':preservation,'reserved_access':False,'qualified_for_final':False}
    atomic_json(R/'reports/bar_independent_validation_v1.json',receipt)
    lines=['# BAR post-result exploratory retrospective results','','This is a new experiment following known SIA results. The historical v2 BAR gate remains unmet and unchanged. No final/P1/Tier-A claims.','','| Track/year | BAR normalized pinball | Strong I0 | SIA v2 | Old RGMF | Own fixed anchor | Gain vs strong |','|---|---:|---:|---:|---:|---:|---:|']
    for key,m in metrics['results'].items():
        x=m['comparisons'];lines.append(f"| {key} | {m['BAR']['normalized_pinball']:.8f} | {x['strong_I0_baseline']['old']['normalized_pinball']:.8f} | {x['SIA_v2']['old']['normalized_pinball']:.8f} | {x['original_RGMF']['old']['normalized_pinball']:.8f} | {x['fixed_BAR_anchor']['old']['normalized_pinball']:.8f} | {x['strong_I0_baseline']['percent_improvement']:.4f}% |")
    lines+=['',f"Decision: {metrics['decision']}. Registered criterion is >=1% normalized loss reduction against both the immutable strongest baseline and its own fixed anchor in both environments, coverage deterioration limits2pp mean/5pp each tail, zero crossings/catastrophes. No expansion automatically follows.",'','Method: width16 source GRU over market+source SIA channels, learned sigmoid reliability/source_count, zero heads and bounded0.15 normalized corrections, exact availability/OOD fallback, ordered quantiles. Main fixed historical and Nested fixed linear-quantile alpha0.1 are refitted past-only for OOF and final anchor. The old strongest Nested alpha0.088586679 remains an immutable separate control. No outer labels trained any model/calibrator.','Inner selection: mature purged13-date validation, epochs0..100 from one fixed optimizer trajectory, then selected-epoch refit on all mature OOF residual data. This is six candidate fits, each with inner training plus optional final refit; selected epoch0 is a legitimate identity-anchor result, not suppressed failure.','OOF support: Main441 dates/inner428, Nested66/inner53. All2496 evaluation rows retained; seeds are not independent market dates. Descriptive block intervals4/8/13 weeks, no confirmatory significance claims.','Calibration: empirical coverage and per-quantile pinball only. No calibrator trained on outer labels. Complete per-seed/per-asset/interval/fallback diagnostics are in bar_exploratory_metrics_v1.json.',f"Budget-free execution: study observed {receipt['observed_study_gpu_wall_seconds']:.6f}s; integration {receipt['observed_integration_gpu_wall_seconds']:.6f}s; independent reload {receipt['observed_validation_gpu_wall_seconds']:.6f}s. These are observations, not charges or ceilings. Integration includes one intentional FAILED fixture; successful/failed records and duplicate rejection preserved.",'All six model bundles independently reloaded, bitwise predictions equal, NumPy scores match, source/policy/config/plan/OOF identities verified. Original357 files/three ledgers and completed v2 outputs/hash unchanged. No reserved data, ABC/HPO/36-fit/v3.2 or strict-PIT/P1/final execution.']
    (R/'reports/V33_BAR_EXPLORATORY_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['configs/bar_post_result_exploratory_v1.json','reports/bar_frozen_plan_v1.json','reports/bar_execution_qualification_v1.json','reports/bar_oof_evidence_v1.json','reports/bar_fits_v1.json','reports/bar_exploratory_metrics_v1.json','reports/bar_independent_validation_v1.json','reports/bar_runner_cuda_integration_v1.json','reports/bar_software_tests_v1.json','reports/V33_BAR_EXPLORATORY_RESULTS.md'];index={p:file_hash(R/p) for p in files};identity=digest(index);folder=T/'artifacts/bar_completion_v1'/identity;commit_bundle(folder,{Path(p).name:(R/p).read_bytes() for p in files},{'plan_id':plan['plan_id'],'external_study_hash':external_hash(),'reserved_access':False});validate_bundle(folder);atomic_json(R/'reports/bar_completion_v1.json',{'completion_id':identity,'relative_bundle':str(folder.relative_to(T)),'files':index,'external_study_hash':external_hash(),'frozen_v2_hash':code_hash(R),'reserved_access':False});progress('BAR_VALIDATED_REPORT_COMPLETE',decision=metrics['decision'],study_seconds=receipt['observed_study_gpu_wall_seconds'],completed_candidates=6)
    return receipt
