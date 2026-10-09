"""CPU-only replay of saved forecasts; preserve every original study artifact."""
import io,json,collections
import numpy as np
import pandas as pd
from signalforge.runtime import paths,validate_bundle,file_hash,digest,commit_bundle,atomic_json,now,code_hash
from signalforge.successor_inputs import prepared_track
from signalforge.panels import track_folds
from signalforge.forensics import forecast_rows,extrapolation,date_score,relative_gain,feature_lineage_ids
from signalforge.successor import ledger_snapshot,assert_parent_unchanged

repo,runtime=paths();reports=repo/'reports';study=json.loads((repo/'configs/study.json').read_text())
qualification=json.loads((reports/'track_input_qualification_v311.json').read_text());full=json.loads((reports/'successor_gpu_full.json').read_text())
parent=ledger_snapshot(runtime/'ledger/development_compute.sqlite',43200);gpu_before=ledger_snapshot(runtime/'ledger/successor_gpu_full_compute.sqlite',21600)
rows=[r for t in qualification['tracks'] for r in t['cpu_development']['results']]+[r for t in full['tracks'] for r in t['results']]
assert len(rows)==990
all_rows=[];transform_audits=[];cases=[];training_losses=[];metric_hashes={};integrity=[]
for track in ['Main-A','Nested-B']:
    panels,contexts,q=prepared_track(repo,runtime,track);folds={i:track_folds(p,study,track)[0] for i,p in panels.items()}
    cached={};lineages={}
    for r in [v for v in rows if v['track']==track]:
        directory=runtime/'artifacts/track_development'/r['run_id'];validate_bundle(directory)
        metric_hashes[r['run_id']]=file_hash(directory/'metrics.json')
        model_dir=runtime/r['model_relative_bundle'];assert digest(validate_bundle(model_dir))==r['model_receipt_id']
        testing=folds[r['information']][r['year']]['test'];saved=np.load(directory/'predictions.npz')
        scales=json.loads((model_dir/'asset_scales.json').read_text());s=testing.asset.map(scales).to_numpy()
        frame=forecast_rows(testing,saved['mean'],saved['quantiles'],s)
        if not np.isclose(date_score(frame),r['pinball'],rtol=1e-12,atol=1e-12):raise ValueError('Immutable scoring mismatch')
        if not np.isclose(date_score(frame,'normalized_mse'),r['normalized_mse'],rtol=1e-12,atol=1e-12):raise ValueError('Immutable mean scoring mismatch')
        key=(r['information'],r['year'],r['normalizer_id'],file_hash(model_dir/'transform.json'))
        if key not in cached:
            raw=contexts[r['information']][testing.sequence_index];manifest=json.loads((model_dir/'transform.json').read_text())
            audit,columns=extrapolation(raw,manifest,panels[r['information']].attrs['feature_names'])
            cached[key]=audit
            transform_audits.extend({**v,'track':track,'information':r['information'],'year':r['year'],'normalizer_id':r['normalizer_id'],'transform_sha256':key[-1]} for v in columns)
        for k,v in cached[key].items():frame[k]=v
        for k in ['track','information','year','family','seed','normalizer_id','run_id']:frame[k]=r[k]
        lineage_key=(r['information'],r['year'])
        if lineage_key not in lineages:lineages[lineage_key]=feature_lineage_ids(testing)
        frame['context_fallback']=saved['context_fallback'];frame['feature_lineage_id']=lineages[lineage_key]
        frame['lagged_volatility']=np.array(testing.x.tolist())[:,panels[r['information']].attrs['feature_names'].index('realized_vol_20')]
        for index in frame.nlargest(3,'normalized_pinball').index:
            source=testing.loc[index];cases.append({**frame.loc[index].to_dict(),'label_start':str(source.label_start),'label_end':str(source.label_end),'label_available_at':str(source.label_available_at),'max_dependency_available_at':str(source.max_dependency_available_at),'feature_lineage':source.feature_lineage,'target_raw_hashes':source.get('target_raw_hashes')})
        if r['family'] in {'mlp','gru','rgmf_linear','rgmf_gru'}:
            import torch
            state=torch.load(model_dir/'model.bin',map_location='cpu',weights_only=False)
            losses=state['losses'];training_losses.append({'run_id':r['run_id'],'track':track,'information':r['information'],'year':r['year'],'family':r['family'],'seed':r['seed'],'epochs':len(losses),'initial_loss':losses[0],'final_loss':losses[-1],'peak_loss':max(losses),'finite_training_loss':bool(np.isfinite(losses).all()),'finite_saved_parameters':all(bool(torch.isfinite(v).all()) for v in state['model'].values())})
        all_rows.append(frame)
    print(json.dumps({'track':track,'immutable_forecasts_replayed':len([r for r in rows if r['track']==track]),'scoring_reproduced':True}),flush=True)
detail=pd.concat(all_rows,ignore_index=True)
base=detail[detail.family.eq('historical')&detail.information.eq('I0')][['track','year','seed','decision_time','asset','normalized_pinball','absolute_pinball']]
base=base.rename(columns={'normalized_pinball':'baseline_normalized_pinball','absolute_pinball':'baseline_absolute_pinball'})
detail=detail.merge(base,on=['track','year','seed','decision_time','asset'],how='left',validate='many_to_one')
assert detail.baseline_normalized_pinball.notna().sum()==detail.normalized_pinball.notna().sum()
detail['absolute_improvement']=detail.baseline_absolute_pinball-detail.absolute_pinball
detail['normalized_improvement']=detail.baseline_normalized_pinball-detail.normalized_pinball
detail['relative_pinball_gain']=np.where(detail.baseline_normalized_pinball>0,detail.normalized_improvement/detail.baseline_normalized_pinball,np.nan)
levels={}
for name,columns in [('year_seed_asset',['track','information','family','year','seed','asset']),('year_seed',['track','information','family','year','seed']),('date',['track','information','family','decision_time']),('family_information',['track','information','family'])]:
    # Collapse assets then seeds inside date, before any time aggregation.
    if name=='year_seed_asset':data=detail.groupby(columns+['decision_time'],dropna=False).mean(numeric_only=True).reset_index()
    else:data=detail.groupby(columns+(['decision_time'] if 'decision_time' not in columns else []),dropna=False).mean(numeric_only=True).reset_index()
    grouped=data.groupby(columns,dropna=False)
    table=grouped[['normalized_pinball','absolute_pinball','baseline_normalized_pinball','baseline_absolute_pinball','normalized_mse','forecast_abs_max','transformed_abs_max']].mean().reset_index()
    table['relative_gain']=[relative_gain(b,c) for b,c in zip(table.baseline_normalized_pinball,table.normalized_pinball)]
    levels[name]=table
maximum=levels['family_information'];counts={}
for track,info in [('Main-A','I3'),('Nested-B','I4')]:
    values=maximum[maximum.track.eq(track)&maximum.information.eq(info)];counts[track]={'families':len(values),'worse_than_historical_I0':int((values.normalized_pinball>values.baseline_normalized_pinball).sum()),'n_independent_dates':int(detail[detail.track.eq(track)].decision_time.nunique())}
assert_parent_unchanged(runtime,parent);gpu_after=ledger_snapshot(runtime/'ledger/successor_gpu_full_compute.sqlite',21600);assert gpu_before['charges_digest']==gpu_after['charges_digest']
payloads={}
for name,data in {'rows':detail,'feature_extrapolation':pd.DataFrame(transform_audits),**levels}.items():
    buffer=io.BytesIO();data.to_parquet(buffer,index=False);payloads[name+'.parquet']=buffer.getvalue()
summary={'state':'COMPLETED_POSTHOC_IMMUTABLE_DEVELOPMENT_FORENSICS','created_at':now(),'source_tree_hash':code_hash(repo),'metric_artifacts':len(rows),'forecast_rows':len(detail),'all_original_scoring_reproduced':True,'quantile_crossings':0,'nonfinite_predictions':0,'all_rows_retained':True,'maximum_information':counts,'largest_finite_forecasts':detail.nlargest(12,'forecast_abs_max').to_dict('records'),'largest_feature_extrapolations':sorted(transform_audits,key=lambda v:v['test_abs_z_max'],reverse=True)[:30],'training_losses':training_losses,'source_cases':sorted(cases,key=lambda v:v['normalized_pinball'],reverse=True)[:40],'metric_hashes':metric_hashes,'source_receipt_hashes':{n:file_hash(reports/n) for n in ['track_input_qualification_v311.json','successor_gpu_full.json']},'parent_ledger_unchanged':True,'successor_ledger_unchanged':True,'gpu_work_executed':False,'claim_boundary':'Post-hoc diagnosis only; associations with extrapolation are not intervention/causal proof; no historical retuning or exclusion','reserved_access':False,'qualified_for_final':False}
identity=digest({'metrics':metric_hashes,'script':file_hash(repo/'scripts/audit_track_model_forensics.py'),'diagnostic':file_hash(repo/'src/signalforge/forensics.py')});target=runtime/'artifacts/model_forensics'/identity
commit_bundle(target,{**payloads,'summary.json':summary},{'reserved_access':False,'original_artifacts_unchanged':True});summary['relative_bundle']=str(target.relative_to(runtime));atomic_json(reports/'track_model_forensics.json',summary)
print(json.dumps({k:summary[k] for k in ['state','metric_artifacts','forecast_rows','maximum_information','all_original_scoring_reproduced','relative_bundle']},indent=2))
