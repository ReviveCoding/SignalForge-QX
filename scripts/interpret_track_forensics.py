"""Read-only original-target/source checks and post-hoc forensic interpretation."""
import json,csv,io
import numpy as np
import pandas as pd
from signalforge.runtime import paths,now,atomic_json,commit_bundle,digest,file_hash,validate_bundle
from signalforge.successor_inputs import prepared_track
from signalforge.track_inputs import read_input_card
from signalforge.forensics import price_mark_index
repo,runtime=paths();summary=json.loads((repo/'reports/track_model_forensics.json').read_text());bundle=runtime/summary['relative_bundle'];validate_bundle(bundle)
detail=pd.read_parquet(bundle/'rows.parquet');families=pd.read_parquet(bundle/'family_information.parquet');yearseed=pd.read_parquet(bundle/'year_seed.parquet')
targets=[];sources=[]
for track in ['Main-A','Nested-B']:
    panels,contexts,qualification=prepared_track(repo,runtime,track);panel=panels['I0']
    manifest=json.loads((repo/'.local'/('inputs_'+track+'_v311.json')).read_text());prices=read_input_card(runtime,manifest['inputs']['prices'])
    marks=price_mark_index(prices)
    checked=0;exceptions=[]
    for row in panel.itertuples():
        if pd.Timestamp(row.decision_time)>=pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError('Reserved target audit forbidden')
        if not np.isfinite(row.y):continue
        left=marks[(row.asset,str(pd.Timestamp(row.label_start)))];right=marks[(row.asset,str(pd.Timestamp(row.label_end)))]
        assert np.isclose(float(right.close)/float(left.close)-1,row.y,rtol=0,atol=1e-12)
        assert sorted(set([left.raw_hash,right.raw_hash]))==row.target_raw_hashes
        if abs(row.y)>1:exceptions.append({'asset':row.asset,'decision_time':str(row.decision_time),'target':row.y,'left_close':left.close,'right_close':right.close,'raw_hashes':row.target_raw_hashes})
        checked+=1
    targets.append({'track':track,'checked_target_rows':checked,'all_raw_close_ratios_match':True,'basis':manifest['p0_basis'],'target_abs_over_one':exceptions,'economics_qualified':False,'asset_ranges':panel.groupby('asset').y.agg(['min','max','std']).reset_index().to_dict('records')})
    for info,p in panels.items():
        dependency=pd.to_datetime(p.max_dependency_available_at,utc=True);origin=pd.to_datetime(p.decision_time,utc=True)
        assert not (dependency>origin).any()
        sources.append({'track':track,'information':info,'rows':len(p),'future_dependency_clocks':0,'pit_tiers':p.attrs['pit_tiers']})
age=detail.dominant_feature.str.contains('release_age',na=False)&detail.dominant_training_scale.eq(1e-8)
explosive=detail.forecast_abs_max.gt(100)
regime=detail.assign(regime=pd.cut(detail.lagged_volatility,[-np.inf,.01,.02,.04,np.inf],labels=['<=1%','1-2%','2-4%','>4%']))
regime_table=regime.groupby(['track','information','family','regime','decision_time'],observed=True)[['normalized_pinball','absolute_pinball']].mean().groupby(level=[0,1,2,3],observed=True).mean().reset_index()
lgb=yearseed[yearseed.family.eq('lightgbm')].copy();base=lgb[lgb.information.eq('I0')][['track','year','seed','normalized_pinball']].rename(columns={'normalized_pinball':'I0_pinball'})
lgb=lgb.merge(base,on=['track','year','seed'],validate='many_to_one');lgb['change_vs_I0']=lgb.normalized_pinball-lgb.I0_pinball
same_family=families.merge(families[families.information.eq('I0')][['track','family','normalized_pinball']].rename(columns={'normalized_pinball':'same_family_I0_pinball'}),on=['track','family'],validate='many_to_one')
increments=[]
for track,info in [('Main-A','I3'),('Nested-B','I4')]:
    selected=same_family[same_family.track.eq(track)&same_family.information.eq(info)].copy()
    selected['relative_gain_vs_same_family_I0']=1-selected.normalized_pinball/selected.same_family_I0_pinball
    increments.append({'track':track,'information':info,'families':len(selected),'worse_than_same_family_I0':int((selected.normalized_pinball>selected.same_family_I0_pinball+1e-12).sum()),'table':selected[['family','normalized_pinball','same_family_I0_pinball','relative_gain_vs_same_family_I0']].to_dict('records')})
evidence=json.loads((repo/'reports/original_archive_evidence_research.json').read_text());validate_bundle(runtime/evidence['relative_bundle'])
notice=next((r for r in evidence['records'] if r['name']=='eia_notice'),None)
result={'state':'COMPLETED_POSTHOC_FORENSIC_INTERPRETATION','created_at':now(),'forensic_receipt_sha256':file_hash(repo/'reports/track_model_forensics.json'),'target_checks':targets,'source_clock_checks':sources,
 'numerical':{'crossings':summary['quantile_crossings'],'nonfinite_predictions':summary['nonfinite_predictions'],'all_original_scores_reproduced':True,'forecast_abs_over_100_rows':int(explosive.sum()),'of_those_dominated_by_training_constant_release_age':int((explosive&age).sum()),'all_forecast_rows_retained':True},
 'diagnosis':{'scoring_defect':'No discrepancy found: original normalization and scores reproduce from saved quantiles and exact fold targets','scaling':'Confirmed constant-column 1e-8 standardization amplifies one-day CFTC release-age shift to 1e8; finite model extrapolation failure, with causal intervention not run','target_basis':'Registered unadjusted P0 close ratios may contain split discontinuities; large true price ratios are retained. These are not total returns or P1 evidence','sources':'Canonical source receipts and per-feature lineage retained, reconstructed clocks <= decision; original versions/first-public clocks remain unqualified','lightgbm':'Macro deterioration and seed/year variation are measured separately below; no favorable seed selection, no causal source-fault conclusion'},
 'lightgbm_year_information_seed':lgb.to_dict('records'),'maximum_information':summary['maximum_information'],'within_family_information_increments':increments,'training_states_finite':all(v['finite_training_loss'] and v['finite_saved_parameters'] for v in summary['training_losses']),
 'eia_disambiguation':{'official_notice_sha256':notice['sha256'] if notice else None,'provider_revision_explanation_verified':bool(notice and notice['checks'].get('provider_explains_mismatch')),'registered_arithmetic_gate_still_blocked':True,'old_failure_receipt_retained':'reports/eia_2023_calibration_source_audit.json','calibration_target_replaced':False},
 'qualified_for_final':False,'reserved_access':False,'claim_boundary':'Post-hoc descriptive diagnosis, not new model training, causal identification or prospective performance'}
buffer=io.BytesIO();regime_table.to_parquet(buffer,index=False);identity=digest(result);target=runtime/'artifacts/forensic_interpretation'/identity
commit_bundle(target,{'interpretation.json':result,'regime.parquet':buffer.getvalue()},{'reserved_access':False,'original_results_unchanged':True});result['relative_bundle']=str(target.relative_to(runtime));atomic_json(repo/'reports/track_model_forensic_interpretation.json',result)
print(json.dumps({k:result[k] for k in ['state','numerical','maximum_information','eia_disambiguation','relative_bundle']},indent=2))
