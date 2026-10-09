"""Second-pass bundle/CUDA replay and scalar independent metric validation."""
import json,io
import numpy as np,pandas as pd,torch
from calibration_v34.runner import *
from diagnostics_v3.core import augment,summary,bootstrap_dates,check_grid

def evaluate():
    from diagnostics_v3.audit import load_forecasts
    c=cfg();cal=load(O/'calibration_frozen.json');assert cal['parameter_id']==digest({k:v for k,v in cal.items() if k!='parameter_id'})
    frozen,folds,oldreceipts=load_forecasts();f=frozen.copy();parts=[]
    for track,year,info in SUBJECTS:
        for seed in [11,37,71]:
            base=f[(f.track==track)&(f.model=='SIA_v2')&(f.seed==seed)].copy();base['model']='SIA_calibrated_v34';base[C]=apply(base[C],base.scale,cal['parameters'][track][str(seed)]);parts.append(base)
    allf=augment(pd.concat([f,*parts],ignore_index=True));allf.to_csv(O/'outer_forecasts.csv',index=False);metrics={};paired={};slices=[];dry=[]
    ex=pd.read_csv(R/'reports/model_risk_v3/ex_ante_features.csv');ex.decision_time=pd.to_datetime(ex.decision_time,utc=True).astype(str);allf.decision_time=pd.to_datetime(allf.decision_time,utc=True).astype(str)
    for track,year,info in SUBJECTS:
        metrics[track]={}
        for model,g in allf[allf.track==track].groupby('model'):
            check_grid(g);metrics[track][model]=summary(g,ci=True)
            for axis in ['seed','asset']:
                for value,h in g.groupby(axis):slices.append({'track':track,'model':model,'slice_axis':axis,'slice':value,**summary(h)})
        candidate=metrics[track]['SIA_calibrated_v34'];paired[track]={}
        for control in ['SIA_v2','strong_I0','original_RGMF']:
            a=allf[(allf.track==track)&(allf.model=='SIA_calibrated_v34')];b=allf[(allf.track==track)&(allf.model==control)];d=a[['decision_time','asset','seed','loss']].merge(b[['decision_time','asset','seed','loss']],on=['decision_time','asset','seed'],suffixes=('_new','_control'),validate='one_to_one');d['delta']=d.loss_new-d.loss_control;paired[track][control]={'absolute_delta':candidate['normalized_pinball']-metrics[track][control]['normalized_pinball'],'relative_improvement_pct':100*(1-candidate['normalized_pinball']/metrics[track][control]['normalized_pinball']),'paired_date_block_uncertainty':bootstrap_dates(d,['delta'])}
        for seed in [11,37,71]:
            h=allf[(allf.track==track)&(allf.model=='SIA_calibrated_v34')&(allf.seed==seed)];dry.append({'track':track,'seed':seed,**monitor(h[C].to_numpy(),h.scale.to_numpy(),cal['thresholds'][track],known_at=h.decision_time.min(),decision_time=h.decision_time.min())})
    # Regimes fixed in previous train-only feature audit, never selected on these outcomes.
    columns=[x for x in ex if x.endswith('_regime')]
    joined=allf.merge(ex[['track','decision_time','asset',*columns]],on=['track','decision_time','asset'],how='left',validate='many_to_one')
    for (track,model),g in joined.groupby(['track','model']):
        for col in columns:
            for value,h in g.groupby(col,dropna=False):slices.append({'track':track,'model':model,'slice_axis':col,'slice':str(value),**summary(h)})
    pd.DataFrame(slices).to_csv(O/'slices.csv',index=False)
    oof=augment(pd.read_csv(O/'genuine_sia_oof.csv'));oofstats={track:summary(g) for track,g in oof.groupby('track')}
    save('evaluation',{'configuration_id':c['configuration_id'],'calibration_id':cal['parameter_id'],'metrics':metrics,'paired':paired,'OOF_metrics':oofstats,'outer_rows_calibrated':len(allf[allf.model=='SIA_calibrated_v34']),'frozen_receipts':oldreceipts,'dry_run_monitor':dry,'exploratory':True,'promotion':False})
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,(track,year,info) in zip(axes,SUBJECTS):
        for model in ['strong_I0','SIA_v2','SIA_calibrated_v34']:ax.plot(Q,metrics[track][model]['coverage'],marker='o',label=model)
        ax.plot(Q,Q,'k--');ax.set(title=track,xlabel='Nominal quantile',ylabel='Empirical coverage');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(O/'calibration_coverage.png',dpi=160);plt.close(fig);event('RETROSPECTIVE_EVALUATION_COMPLETE',metrics={t:{m:s['normalized_pinball'] for m,s in v.items()} for t,v in metrics.items()})

def validate():
    c=cfg();preservation=preserve();fits=load(O/'oof_completion.json')['results'];prepared=load(O/'prepared.json');assert len(fits)==len(load(O/'plan.json')['fits']);reloads=[];meter=ObservationMeter(T/'ledger/v33_budget_free_v3_sia_oof_v34_validation.sqlite')
    try:
        with device_lease(2*1024**3):
            for track,year,info in SUBJECTS:
                tr,ctx,schema,contract,source,cut=cohort(track,year,info);parts=mature_partitions(tr.decision_time,tr.label_end,tr.label_available_at,minimum=26,block=13)
                for i,p in enumerate(parts):
                    a,m,h=block_input(tr,ctx,schema,contract,p)
                    for r in [r for r in fits if r['track']==track and r['block']==i+1]:
                        b=T/r['relative_bundle'];validate_bundle(b);assert r['checkpoint_hash']==file_hash(r['checkpoint_path']);ck=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False);assert ck['step']==100 and ck['sample_cursor']==100*len(a['y']) and len(ck['losses'])==100
                        record=load(Path(r['checkpoint_path']).with_suffix('.receipt.json'));assert record['identity']==ck['identity'] and record['sha256']==r['checkpoint_hash'];assert m['array_hashes']==r['input_metadata']['array_hashes']
                        identity=digest({'validation_fit':r['identity'],'configuration_id':c['configuration_id']})
                        if not any(x['identity']==identity for x in meter.rows()):
                            with meter.attempt(identity,metadata={'phase':'independent reload only','fit':r['identity']}) as observed:
                                torch.cuda.reset_peak_memory_stats();model=ExplicitSourceCUDA.load(b/'model.pt');q=model.predict(a['vx'],observed=a['test_observed'],source_valid=a['test_valid'])[1]*a['scale'][:,None];assert np.array_equal(q,np.load(b/'predictions.npz')['quantiles']);observed.observe(peak_vram_bytes=int(torch.cuda.max_memory_allocated()),outcome={'bitwise_equal':True});del model;torch.cuda.empty_cache()
                        reloads.append(r['identity'])
                    event('INDEPENDENT_CUDA_REPLAY_BLOCK',track=track,block=i+1,replayed=len(reloads))
        records=meter.rows()
    finally:meter.close()
    oof=pd.read_csv(O/'genuine_sia_oof.csv');assert not oof.duplicated(['track','decision_time','asset','seed']).any();assert len(oof)==sum(r['rows'] for r in fits)
    for track,year,info in SUBJECTS:
        g=oof[oof.track==track];assert set(g.seed)=={11,37,71}
        for _,h in g.groupby('decision_time'):assert len(h)==24 and h.asset.nunique()==8
        assert (np.diff(g[C],axis=1)>=0).all() and np.isfinite(g[C]).all().all()
    f=pd.read_csv(O/'outer_forecasts.csv');ev=load(O/'evaluation.json');independent={}
    for (track,model),g in f.groupby(['track','model']):
        scalar=[]
        for row in g.itertuples():
            values=[max(float(t)*(row.target-getattr(row,col)),(float(t)-1)*(row.target-getattr(row,col)))/row.scale for t,col in zip(Q,C)];scalar.append(sum(values)/5)
        recompute=pd.DataFrame({'d':g.decision_time.to_numpy(),'v':scalar}).groupby('d').v.mean().mean();assert np.isclose(recompute,ev['metrics'][track][model]['normalized_pinball'],rtol=0,atol=2e-14);independent[track+':'+model]=float(recompute)
    assert len(f[f.model=='SIA_calibrated_v34'])==2496
    for path,sha in ev['frozen_receipts'].items():assert file_hash(path)==sha
    cal=load(O/'calibration_frozen.json');assert file_hash(O/'genuine_sia_oof.csv')==cal['OOF_hash'];assert cal['parameter_id']==digest({k:v for k,v in cal.items() if k!='parameter_id'})
    tests();save('independent_validation',{'passed':True,'configuration_id':c['configuration_id'],'source_hash':source_hash(),'CUDA_replay_count':len(reloads),'OOF_rows':len(oof),'outer_rows':2496,'independent_scalar_metrics':independent,'checkpoint_valid':True,'validation_meter':records,'preservation':preserve(),'reserved_access':False,'final_access':False})
    study=load(O/'oof_completion.json')['meter_records'];seconds=sum(r['elapsed_seconds'] for r in study);ev=load(O/'evaluation.json');text=['# V34 genuine mature SIA OOF and calibration','', 'Post-result exploratory development. No independent future confirmation, P1, Tier-A or final claim.',f'Genuine SIA CUDA OOF fits: {len(fits)}; measured study wall time while GPU leased: {seconds:.6f} seconds. Two additional real CUDA integration fits; 120 independent reload checks (no training).',f'OOF rows: {len(oof)}. Outer calibrated rows: 2496, all 52 dates × 8 assets × 3 seeds per track retained.','', '| Track | Strong I0 | Frozen SIA | Calibrated SIA | 90% coverage before → after |','|---|---:|---:|---:|---:|']
    for track,year,info in SUBJECTS:
        m=ev['metrics'][track];text.append(f"| {track} | {m['strong_I0']['normalized_pinball']:.9f} | {m['SIA_v2']['normalized_pinball']:.9f} | {m['SIA_calibrated_v34']['normalized_pinball']:.9f} | {m['SIA_v2']['coverage90']:.6f} → {m['SIA_calibrated_v34']['coverage90']:.6f} |")
    text+=['','## Methods and limits','Expanding prequential blocks require at least 26 mature past dates; each block predicts the next 13 dates, with no held/outer labels in model training or preprocessing. Three fixed seeds, width 16, lr .001, 100 epochs; faithful frozen SIA adapter and source masks. Each block has its own train-only feature transform and asset scale. Predictions and calibrator offsets are retained in native units; normalized offsets use each block training scale and outer frozen training scale. No per-asset tail fitting or seed selection.','Chronological inner last-13-date calibration selection uses maturity-purged OOF prefix, date/asset equal and seed-average loss. Identity/intercept/CQR choices frozen before new outer access. Unsupported quantiles remain exact identity, including projection anchors. Tail support additionally requires ≥20 expected smaller-tail independent dates. Nested-B tails are underpowered and remain identity. CQR has no IID or guaranteed coverage claim under dependent financial observations.','Prior train-only ex-ante regime slices are reused unchanged. Per-seed/asset/regime, all five quantile losses/coverages, 80/90 interval score/width, concentration and date-block 4/8/13 uncertainty are in evaluation.json and slices.csv. These are descriptive, not confirmatory confidence or candidate promotion.','OOF-only risk thresholds and callable monitor are candidates, not operationally frozen or live notifications. No BAR recalibration, HPO, reserved access or final access. Model/scaler/normalizer/checkpoint lineage and hashes are in fit bundles, prepared.json and plan.json.','Independent validation replays every model on actual CUDA with bitwise saved prediction equality and scalar-recomputes all paired outer losses. Original 357 files/three ledgers, frozen v2 hash, completed SIA/BAR receipts preserved. Synthetic regression tests are distinct from real research evidence.','See calibration_frozen.json for each seed parameters, exact inner support and selected candidate; genuine_sia_oof.csv for every prediction; independent_validation.json for acceptance.']
    (O/'V34_SIA_MATURE_OOF_AND_CALIBRATION.md').write_text('\n'.join(text)+'\n');event('VALIDATED_CALIBRATION_COMPLETE',successful_fits=len(fits),OOF_rows=len(oof),outer_rows=2496,measured_study_seconds=seconds)
    index={str(p.relative_to(R)):file_hash(p) for p in sorted(O.iterdir()) if p.is_file() and p.name not in ['handoff.json','status.json','STATUS.json']};bundle=T/'artifacts/sia_oof_v34/handoff'/digest(index);commit_bundle(bundle,{p: (R/p).read_bytes() for p in index},{'configuration_id':c['configuration_id'],'reserved_access':False});save('handoff',{'files':index,'relative_bundle':str(bundle.relative_to(T)),'configuration_id':c['configuration_id'],'v2_hash':code_hash(R),'implementation_scope':'completed mature OOF/calibration exploratory; external scientific gates blocked'})