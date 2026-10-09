"""Executed post-result BAR study, opt-in budget-free; outside frozen v2 source."""
import argparse,io,json,os,sys,time,traceback,subprocess,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from signalforge.runtime import code_hash,digest,file_hash,atomic_json,commit_bundle,validate_bundle,now
from signalforge.successor import ledger_snapshot
from signalforge.track_engine import prepare,per_date_weights,asset_normalizer
from signalforge.track_statistics import DateWeightedStatistical
from signalforge.panels import track_folds
from signalforge.v33_sia import prepare_sia,schema_from_specs
from signalforge.v33_corrections import mature_partitions
from signalforge.v33_comparison import paired_comparison,COLS,Q
from signalforge.checkpoint import save_checkpoint,load_checkpoint,restore_rng
from policy_v3.v33_budget_free import ObservationMeter,device_lease,admission,policy,policy_source_hash,CooperativeCancellation
from bar_model import BAR,source_ood,source_tensors,native_predict
from build_v33_readiness import load_original,ORIGINAL,ORIGINAL_RUNTIME,bytes_hash

R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev')
T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering')
PROTOCOL='sgqx-v33-BAR-post-result-exploratory-v1r1'
V2HASH='471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870'
CONFIG=R/'configs/bar_post_result_exploratory_v1r1.json'
PLAN=R/'reports/bar_frozen_plan_v1.json'
METER=T/'ledger/v33_budget_free_v3_bar_study_v1.sqlite'
IMETER=T/'ledger/v33_budget_free_v3_bar_integration_v1.sqlite'

def iso_clock(value):return pd.Timestamp(value).isoformat()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def external_hash():return digest({str(p.relative_to(R)):file_hash(p) for p in sorted((R/'studies_v3').glob('*')) if p.suffix in {'.py','.ps1'}})
def progress(stage,**fields):
    row={'created_at':now(),'stage':stage,'protocol_id':PROTOCOL,'external_study_hash':external_hash(),'reserved_access':False,**fields};atomic_json(R/'reports/bar_live_progress_v1.json',row);print(json.dumps(row),flush=True)
def intact():
    assert code_hash(R)==V2HASH
    baseline=load(R/'reports/v33_handoff_reconfirmation.json');assert code_hash(ORIGINAL)==baseline['original_source_hash']
    for name,expected in baseline['original_ledgers'].items():
        actual=ledger_snapshot(ORIGINAL_RUNTIME/'ledger'/name,expected['ceiling_seconds']);actual.pop('charges');assert actual==expected
    c=load(R/'reports/v33_minimal_completion_v2.json');validate_bundle(T/c['relative_bundle'])
    for p,h in c['files'].items():assert file_hash(R/p)==h
    for row in load(R/'reports/v33_minimal_study_v2.json')['results']:validate_bundle(T/row['relative_bundle'])
    return {'frozen_v2_hash':V2HASH,'original_source_hash':baseline['original_source_hash'],'original_ledgers':baseline['original_ledgers'],'all_completed_v2_receipts_and_models_valid':True}
def study():
    c=load(CONFIG);assert c['external_study_hash']==external_hash() and c['policy_source_hash']==policy_source_hash() and c['frozen_v2_hash']==code_hash(R)
    assert c['configuration_id']==digest({k:v for k,v in c.items() if k!='configuration_id'});return c

def tests():
    p=subprocess.run([sys.executable,'-m','pytest','studies_v3/test_bar_study.py','tests/v33/test_budget_free_policy.py','-q','-p','no:cacheprovider'],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(R)+':'+str(R/'src')))
    print(p.stdout);print(p.stderr);atomic_json(R/'reports/bar_software_tests_v1.json',{'exit_code':p.returncode,'log':p.stdout+p.stderr,'external_study_hash':external_hash(),'policy_source_hash':policy_source_hash(),'frozen_v2_hash':code_hash(R),'scope':'CPU engineering fixtures, not trained research','reserved_access':False});assert p.returncode==0

def integration():
    intact();test=load(R/'reports/bar_software_tests_v1.json');assert test['exit_code']==0 and test['external_study_hash']==external_hash()
    meter=ObservationMeter(IMETER);success=[]
    try:
        with device_lease(2*1024**3):
            for track,year in [('Main-A',2020),('Nested-B',2022)]:
                v2=load(R/'configs/v33_minimal_execution_v2r1.json');b=T/v2['data_bundles'][track+':'+str(year)];validate_bundle(b);z=np.load(b/'arrays.npz');meta=load(b/'data.json');groups=[meta['neural_args']['base_columns']+s for s in meta['neural_args']['source_columns']]
                identity=digest({'phase':'real_cuda_integration','track':track,'external_hash':external_hash(),'policy_hash':policy_source_hash()});target=T/'artifacts/bar_integration_v1'/identity
                if (target/'receipt.json').exists():validate_bundle(target);success.append(load(target/'integration.json'));continue
                progress('BUDGET_FREE_CUDA_INTEGRATION',track=track,year=year)
                with meter.attempt(identity,metadata={'purpose':'engineering integration only','track':track,'study_hash':external_hash()}) as observed:
                    torch.manual_seed(11);model=BAR([len(g) for g in groups],width=16).cuda();torch.cuda.reset_peak_memory_stats();optim=torch.optim.AdamW(model.parameters(),lr=.001)
                    x=source_tensors(z['tx'][:32],groups);valid=torch.as_tensor(z['train_valid'][:32],device='cuda');ood=torch.zeros_like(valid);base=torch.tensor(np.tile([-2.,-1.,0.,1.,2.],(32,1)),device='cuda');y=torch.as_tensor(z['y'][:32],device='cuda')
                    for epoch in range(2):
                        optim.zero_grad();q=model(base,x,valid,ood);delta=y[:,None]-q;loss=torch.maximum(torch.as_tensor(Q,device='cuda')*delta,(torch.as_tensor(Q,device='cuda')-1)*delta).mean();loss.backward();optim.step();observed.observe(peak_vram_bytes=int(torch.cuda.max_memory_allocated()))
                    base_native=z['strong_baseline_11'];q,fb=native_predict(model,z['vx'],groups,base_native,z['scale'],z['test_valid'],np.zeros_like(z['test_valid']))
                    with tempfile.TemporaryDirectory(dir=T/'tmp') as tmp:
                        path=Path(tmp)/'model.pt';torch.save({'spec':model.specification(),'state':model.state_dict()},path);state=torch.load(path,map_location='cuda',weights_only=False);copy=BAR(**{'dimensions':state['spec']['dimensions'],'width':state['spec']['width'],'bound':state['spec']['bound']}).cuda();copy.load_state_dict(state['state']);rq,_=native_predict(copy,z['vx'],groups,base_native,z['scale'],z['test_valid'],np.zeros_like(z['test_valid']));assert np.array_equal(q,rq)
                        missing,_=native_predict(copy,z['vx'],groups,base_native,z['scale'],np.zeros_like(z['test_valid']),np.zeros_like(z['test_valid']));assert np.array_equal(missing,base_native)
                        evidence={'track':track,'year':year,'fit_predict_reload_identical':True,'native_exact_fallback':True,'actual_device':torch.cuda.get_device_name(),'external_study_hash':external_hash(),'policy_source_hash':policy_source_hash(),'input_dimensions':list(z['tx'].shape),'label_maturity_provenance':meta['label_maturity_verified'],'purpose':'real CUDA engineering qualification; fixed toy training anchors, real mature inputs/targets; not performance evidence'}
                        commit_bundle(target,{'model.pt':path.read_bytes(),'integration.json':evidence},{'study_hash':external_hash(),'reserved_access':False});success.append(evidence)
                    observed.observe(outcome={'reload_identical':True,'exact_fallback':True});del model,copy;torch.cuda.empty_cache()
                try:
                    with meter.attempt(identity):pass
                except ValueError:pass
                else:raise AssertionError('Duplicate meter identity not rejected')
            fixture_id=digest({'phase':'intentional_failure_fixture','study_hash':external_hash()})
            if not any(r['identity']==fixture_id for r in meter.rows()):
                try:
                    with meter.attempt(fixture_id,metadata={'purpose':'intentional integration failure fixture, not research fit'}):raise RuntimeError('Intentional test failure')
                except RuntimeError:pass
            records=meter.rows();assert any(r['state']=='FAILED' and r['identity']==fixture_id for r in records)
        receipt={'passed':len(success)==2,'phase':'budget-free actual runner integration','external_study_hash':external_hash(),'policy_source_hash':policy_source_hash(),'frozen_v2_hash':code_hash(R),'success':success,'meter_records':records,'no_time_allowance_or_cost_approval':True,'failure_fixture_retained':True,'duplicate_id_rejected':True,'reserved_access':False};atomic_json(R/'reports/bar_runner_cuda_integration_v1.json',receipt);progress('RUNNER_INTEGRATION_PASSED',successful_cuda_checks=2,meter_records=len(records))
    finally:meter.close()

def register():
    intact();q=load(R/'reports/bar_runner_cuda_integration_v1.json');assert q['passed'] and q['external_study_hash']==external_hash()
    v1=load(R/'configs/v33_minimal_plan_v1.json')
    c={'protocol_id':PROTOCOL,'scope':'POST_RESULT_EXPLORATORY_RETROSPECTIVE_ONLY','frozen_v2_hash':V2HASH,'external_study_hash':external_hash(),'policy_id':policy()['policy_id'],'policy_source_hash':policy_source_hash(),'policy_hash':digest(policy()),'architecture':{'source_encoders':'GRU per market+source sequence','width':16,'zero_init_residual_heads':True,'normalized_residual_bound':.15,'gate':'sigmoid(source latent)/source_count with raw availability and any-context numeric OOD mask','non_crossing':'predeclared cumulative-max projection','exact_native_fallback':True},'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'max_inner_epochs':100,'clip_gradient_norm':1.,'seed_rule':[11,37,71],'epoch_selection':'minimum mature inner normalized pinball including zero-epoch anchor; earliest tie; final refit selected epoch count'},'baseline_rule':{'Main-A':{'family':'historical','regularization':1.},'Nested-B':{'family':'linear_quantile','regularization':.1}},'baseline_scope':'Fixed new rule before new BAR outcomes; alpha0.1 is not copied from future old fold tuning. Exact old strongest baseline remains immutable comparison control. Compare own fixed anchor too.','OOF':{'minimum_past_mature_dates':26,'prediction_block_dates':13,'minimum_residual_dates':26,'inner_validation_dates':13,'inner_past_mature_minimum':26,'folds':'original exact common-context training grid, same target-normalizer IDs','baseline_preprocessing':'per-block past-only original market-only transformer/asset scale, native-unit predictions','residual_preprocessing':'typed SIA fitted on maturity-qualified prefix for inner fit, full mature training for final refit'},'fits':[dict(row,family='bar_gru',settings={'width':16,'lr':.001,'bound':.15,'max_inner_epochs':100}) for row in v1['fits']],'folds':v1['folds'],'decision_rule':{'normalized_pinball_lower_than_strong_control_both_tracks':True,'normalized_pinball_lower_than_own_anchor_both_tracks':True,'minimum_relative_improvement':.01,'coverage_mean_absolute_error_max_deterioration_vs_strong':.02,'each_tail_coverage_error_max_deterioration_vs_strong':.05,'crossings_and_abs_forecast_over100':0,'otherwise':'negative/inconclusive; stop expansion'},'uncertainty':{'date_blocks':[4,8,13],'draws':2000,'seed':20261008,'claim':'descriptive moving-block bootstrap intervals only; no significance/confirmatory claims'},'old_BAR_gate_not_reinterpreted':True,'HPO_or_ABC_or_v32_authorized':False,'reserved_access':False,'final_access':False}
    c['configuration_id']=digest(c)
    if CONFIG.exists():assert load(CONFIG)==c,'Frozen configuration changed'
    else:atomic_json(CONFIG,c)
    progress('NEW_BAR_PROTOCOL_REGISTERED',configuration_id=c['configuration_id'],post_result_exploratory=True,planned_candidates=6)

def encode_part(train,test,contexts,cutoff,schema,contract):
    result=prepare_sia(train,test,contexts,cutoff,schema,neural_contract=contract)
    groups=[result['neural_args']['base_columns']+s for s in result['neural_args']['source_columns']]
    return result,groups,source_ood(result['train'],contract['source_raw_columns']),source_ood(result['test'],contract['source_raw_columns'])

def prepare_data():
    c=study();intact();bundles={};fold_metadata={};registered_fits=[];OOF_reports=[]
    for track,year,info in [('Main-A',2020,'I3'),('Nested-B',2022,'I4')]:
        progress('PREPARING_MATURE_PAST_ONLY_OOF',track=track,year=year)
        manifest,settings,panels,contexts,source=load_original(track);fs={i:track_folds(p,settings,track)[0][year] for i,p in panels.items()};common=set.intersection(*[set(zip(f['train'].loc[f['train'].context_eligible,'decision_time'],f['train'].loc[f['train'].context_eligible,'asset'])) for f in fs.values()]);train=fs[info]['train'];train=train[[k in common for k in zip(train.decision_time,train.asset)]].sort_values(['decision_time','asset']);test=fs[info]['test'];cutoff=fs[info]['outer_cutoff'];baseframe=fs['I0']['train'];baseframe=baseframe[[k in common for k in zip(baseframe.decision_time,baseframe.asset)]].sort_values(['decision_time','asset'])
        assert list(zip(train.decision_time,train.asset))==list(zip(baseframe.decision_time,baseframe.asset))
        specs=[s for s in manifest['features'] if s['source'] in settings['information_sets'][info]];schema=schema_from_specs(specs,panels[info].attrs['feature_names']);contract=panels[info].attrs['neural_contract']
        full,groups,trainood,testood=encode_part(train,test,contexts[info],cutoff,schema,contract)
        frozen=c['folds'][track+':'+str(year)];assert full['normalizer_id']==frozen['normalizer_id'] and full['transform'].manifest()['manifest_id']==frozen['sia_manifest_id']
        base_rule=c['baseline_rule'][track];parts=mature_partitions(train.decision_time,train.label_end,train.label_available_at,minimum=26,block=13);oof=np.full((len(train),5),np.nan);traces=[]
        for number,p in enumerate(parts):
            past=baseframe.iloc[p['train']];held=baseframe.iloc[p['oof']];prepared=prepare(past,held,contexts['I0'],pd.Timestamp(p['cutoff']))
            model=DateWeightedStatistical(base_rule['family'],base_rule['regularization']).fit(prepared['tx'],prepared['y'],weights=prepared['weights']);q=model.predict(prepared['vx'])[1]*prepared['test_scale'][:,None];oof[p['oof']]=q
            trace={'cutoff':p['cutoff'],'past_dates':past.decision_time.nunique(),'max_label_available_at':str(past.label_available_at.max()),'past_keys_hash':digest(past[['decision_time','asset']].astype(str).values.tolist()),'predicted_keys_hash':digest(held[['decision_time','asset']].astype(str).values.tolist()),'normalizer_id':prepared['normalizer_id'],'transform_id':digest(prepared['transform'].manifest())};traces.append(trace)
            progress('MATURE_OOF_BLOCK_SAVED',track=track,year=year,block=number+1,blocks=len(parts),past_dates=trace['past_dates'])
        usable=np.isfinite(oof).all(axis=1);unique=train.loc[usable,'decision_time'].unique();assert len(unique)>=26+13
        innercut=pd.Timestamp(unique[-13]);inner_train=(train.decision_time<innercut)&(train.label_end<=innercut)&(train.label_available_at<=innercut);validation=usable&train.decision_time.isin(unique[-13:]);residual_inner=usable&inner_train.to_numpy()
        assert train.loc[residual_inner,'decision_time'].nunique()>=26 and train.loc[validation,'decision_time'].nunique()==13
        inner,igroups,innerood,validationood=encode_part(train.loc[inner_train],train.loc[validation],contexts[info],innercut,schema,contract)
        # Restrict residual labels to OOF rows but fit normalization on all past mature prefix.
        inner_positions=train.loc[inner_train].index.get_indexer(train.loc[residual_inner].index);assert (inner_positions>=0).all();inner_scale=train.loc[residual_inner,'asset'].map(inner['asset_scales']).to_numpy();valscale=train.loc[validation,'asset'].map(inner['asset_scales']).to_numpy();fullscale=train.loc[usable,'asset'].map(full['asset_scales']).to_numpy()
        anchor_prepare=prepare(baseframe,test,contexts['I0'],cutoff);assert anchor_prepare['normalizer_id']==full['normalizer_id'];anchor=DateWeightedStatistical(base_rule['family'],base_rule['regularization']).fit(anchor_prepare['tx'],anchor_prepare['y'],weights=anchor_prepare['weights']);outer_base=anchor.predict(anchor_prepare['vx'])[1]*full['test_scale'][:,None]
        arrays={'inner_x':inner['train']['model_matrix'][inner_positions],'inner_y':train.loc[residual_inner,'y'].to_numpy()/inner_scale,'inner_base':oof[residual_inner]/inner_scale[:,None],'inner_valid':inner['source_valid'][inner_positions],'inner_ood':innerood[inner_positions],'inner_weights':per_date_weights(train.loc[residual_inner]),'val_x':inner['test']['model_matrix'],'val_y':train.loc[validation,'y'].to_numpy()/valscale,'val_base':oof[validation]/valscale[:,None],'val_valid':inner['test_source_valid'],'val_ood':validationood,'val_weights':per_date_weights(train.loc[validation]),'train_x':full['train']['model_matrix'][usable],'train_y':train.loc[usable,'y'].to_numpy()/fullscale,'train_base':oof[usable]/fullscale[:,None],'train_valid':full['source_valid'][usable],'train_ood':trainood[usable],'train_weights':per_date_weights(train.loc[usable]),'test_x':full['test']['model_matrix'],'test_base':outer_base,'test_valid':full['test_source_valid'],'test_ood':testood,'target':test.y.to_numpy(),'scale':full['test_scale'],'oof_native':oof}
        meta={'track':track,'year':year,'information':info,'fold_id':frozen['fold_id'],'source_id':frozen['source_id'],'normalizer_id':frozen['normalizer_id'],'schema_id':full['transform'].manifest()['manifest_id'],'target_id':bytes_hash(test.y),'groups':groups,'inner_groups':igroups,'OOF_dates':len(unique),'inner_residual_dates':train.loc[residual_inner,'decision_time'].nunique(),'inner_validation_dates':13,'fit_cutoff':cutoff.isoformat(),'training_decision_times':[iso_clock(d) for d in train.decision_time],'label_end':[iso_clock(d) for d in train.label_end],'label_available_at':[iso_clock(d) for d in train.label_available_at],'date_asset_keys':[[iso_clock(d),a] for d,a in zip(test.decision_time,test.asset)],'assets':sorted(test.asset.unique()),'OOF_traces':traces,'inner_cutoff':innercut.isoformat(),'full_transform':full['transform'].manifest(),'inner_transform':inner['transform'].manifest(),'context_eligible_all':bool(test.context_eligible.all()),'source_ood_test_rows':int(testood.any(1).sum())}
        assert meta['context_eligible_all'];meta['date_asset_keys_hash']=digest(meta['date_asset_keys']);identity=digest({'configuration_id':c['configuration_id'],'metadata':meta,'array_hashes':{k:bytes_hash(v) for k,v in arrays.items()}});dest=T/'artifacts/bar_prepared_v1'/identity;stream=io.BytesIO();np.savez_compressed(stream,**arrays);commit_bundle(dest,{'arrays.npz':stream.getvalue(),'metadata.json':meta},{'configuration_id':c['configuration_id'],'reserved_access':False})
        bundles[track+':'+str(year)]=str(dest.relative_to(T));fold_metadata[meta['fold_id']]={k:meta[k] for k in ['source_id','schema_id','normalizer_id','target_id','date_asset_keys_hash','date_asset_keys','assets','fit_cutoff','training_decision_times','label_end','label_available_at']}
        for seed in [11,37,71]:registered_fits.append({'fold_id':meta['fold_id'],'source_id':meta['source_id'],'schema_id':meta['schema_id'],'normalizer_id':meta['normalizer_id'],'target_id':meta['target_id'],'date_asset_keys_hash':meta['date_asset_keys_hash'],'seed':seed})
        OOF_reports.append({'track':track,'year':year,'OOF_dates':meta['OOF_dates'],'inner_dates':meta['inner_residual_dates'],'blocks':len(parts),'OOF_id':bytes_hash(oof),'relative_bundle':str(dest.relative_to(T))})
    qualification={'source_tree_hash':V2HASH,'policy_source_hash':policy_source_hash(),'passed':True,'cuda_inputs_validated':True,'plan_contract_id':c['configuration_id'],'external_study_hash':external_hash(),'folds':fold_metadata,'validated_fits':registered_fits,'real_cuda_integration_receipt_sha256':file_hash(R/'reports/bar_runner_cuda_integration_v1.json')};qualification['qualification_id']=digest(qualification)
    plan={'policy_id':policy()['policy_id'],'policy_hash':digest(policy()),'policy_source_hash':policy_source_hash(),'source_tree_hash':V2HASH,'external_study_hash':external_hash(),'plan_contract_id':c['configuration_id'],'qualification_id':qualification['qualification_id'],'local_only':True,'reserved_access':False,'final_access':False,'fits':registered_fits,'data_bundles':bundles};plan['plan_id']=digest(plan);admitted=admission(plan,qualification,experiment_requested=True)
    atomic_json(R/'reports/bar_execution_qualification_v1.json',qualification);atomic_json(PLAN,plan);atomic_json(R/'reports/bar_oof_evidence_v1.json',{'OOF_reports':OOF_reports,'admission':admitted,'reserved_access':False});progress('BAR_SCIENTIFIC_AND_DEVICE_ADMISSION_READY',planned_candidates=6,OOF_reports=OOF_reports)

def tensor_data(z,prefix,groups):
    return (torch.as_tensor(z[prefix+'_base'],dtype=torch.float64,device='cuda'),source_tensors(z[prefix+'_x'],groups),torch.as_tensor(z[prefix+'_valid'],device='cuda'),torch.as_tensor(z[prefix+'_ood'],device='cuda'),torch.as_tensor(z[prefix+'_y'],dtype=torch.float64,device='cuda'),torch.as_tensor(z[prefix+'_weights'],dtype=torch.float64,device='cuda'))
def loss(model,args):
    base,x,valid,ood,y,w=args;q=model(base,x,valid,ood);delta=y[:,None]-q;levels=torch.as_tensor(Q,device='cuda');pin=torch.maximum(levels*delta,(levels-1)*delta).mean(1);return (pin*w).sum()/w.sum()
def optimize(z,groups,seed,epochs,phase,identity,observer,*,validation=False):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=BAR([len(g) for g in groups],width=16,bound=.15).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);args=tensor_data(z,'inner' if validation else 'train',groups);val=tensor_data(z,'val',groups) if validation else None;checkpoint=T/'checkpoints/bar_v1'/(identity+'_'+phase+'.pt');history=[];best_epoch=0
    best=float(loss(model,val).detach()) if validation else None;start=0
    state=load_checkpoint(checkpoint,identity+':'+phase)
    if state is not None:
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state);start=state['step'];history=state['history'];best=state['best'];best_epoch=state['best_epoch'];assert state['sample_cursor']==start*len(args[4])
    for epoch in range(start,epochs):
        if (R/'.local/STOP_BAR_EXPLORATORY').exists() or (R/'.local/STOP_LOCAL_GPU').exists():observer.cancel()
        model.train();optimizer.zero_grad(set_to_none=True);objective=loss(model,args)
        if not torch.isfinite(objective):raise RuntimeError('Nonfinite BAR loss')
        objective.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite BAR gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();score=None
        if validation:
            model.eval()
            with torch.no_grad():score=float(loss(model,val))
            if score<best:best=score;best_epoch=epoch+1
        history.append({'epoch':epoch+1,'train_pinball':float(objective.detach()),'inner_pinball':score});observer.observe(peak_vram_bytes=int(torch.cuda.max_memory_allocated()),outcome={'phase':phase,'epoch':epoch+1,'checkpoint':str(checkpoint.relative_to(T))})
        if (epoch+1)%10==0 or epoch+1==epochs:save_checkpoint(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':None,'step':epoch+1,'sample_cursor':(epoch+1)*len(args[4]),'history':history,'best':best,'best_epoch':best_epoch},identity+':'+phase)
        if (epoch+1)%25==0:progress('BAR_CUDA_EPOCH',fit_id=identity,phase=phase,epoch=epoch+1,planned_epochs=epochs)
    model.eval();return model,best_epoch,history

def run():
    c=study();plan=load(PLAN);q=load(R/'reports/bar_execution_qualification_v1.json');assert plan['external_study_hash']==external_hash();admission(plan,q,experiment_requested=True);intact();meter=ObservationMeter(METER);completed=[]
    try:
        with device_lease(3*1024**3):
            for track,year in [('Main-A',2020),('Nested-B',2022)]:
                b=T/plan['data_bundles'][track+':'+str(year)];validate_bundle(b);z=np.load(b/'arrays.npz');meta=load(b/'metadata.json')
                for seed in [11,37,71]:
                    identity=digest({'plan_id':plan['plan_id'],'track':track,'year':year,'seed':seed,'external_study_hash':external_hash()});dest=T/'artifacts/bar_models_v1'/identity
                    if (dest/'receipt.json').exists():validate_bundle(dest);completed.append(load(dest/'fit.json'));continue
                    existing=[x for x in meter.rows() if x['metadata'] and json.loads(x['metadata']).get('fit_id')==identity];retry_of=existing[-1]['identity'] if existing else None;attempt_id=identity if not existing else identity+':retry'+str(len(existing));progress('BAR_CANDIDATE_START',track=track,year=year,seed=seed,completed_candidates=len(completed),planned_candidates=6)
                    try:
                        with meter.attempt(attempt_id,retry_of=retry_of,metadata={'fit_id':identity,'track':track,'year':year,'seed':seed,'plan_id':plan['plan_id'],'external_study_hash':external_hash(),'checkpoint_directory':'checkpoints/bar_v1'}) as observed:
                            torch.cuda.reset_peak_memory_stats();inner,best,innerhistory=optimize(z,meta['inner_groups'],seed,100,'inner',identity,observed,validation=True);del inner;torch.cuda.empty_cache();final,_,history=optimize(z,meta['groups'],seed,best,'refit',identity,observed)
                            prediction,fallback=native_predict(final,z['test_x'],meta['groups'],z['test_base'],z['scale'],z['test_valid'],z['test_ood'])
                            with tempfile.TemporaryDirectory(dir=T/'tmp') as tmp:
                                path=Path(tmp)/'model.pt';torch.save({'spec':final.specification(),'state':final.state_dict(),'selected_epoch':best},path);saved=torch.load(path,map_location='cuda',weights_only=False);again=BAR(**saved['spec']).cuda();again.load_state_dict(saved['state']);reload,refallback=native_predict(again,z['test_x'],meta['groups'],z['test_base'],z['scale'],z['test_valid'],z['test_ood']);assert np.array_equal(prediction,reload) and np.array_equal(fallback,refallback)
                                absent,_=native_predict(again,z['test_x'],meta['groups'],z['test_base'],z['scale'],np.zeros_like(z['test_valid']),z['test_ood']);assert np.array_equal(absent,z['test_base'])
                                stream=io.BytesIO();np.savez(stream,quantiles=prediction,fallback=fallback)
                                fit={'state':'SUCCEEDED_EXPLORATORY_CANDIDATE','fit_id':identity,'attempt_id':attempt_id,'track':track,'year':year,'seed':seed,'selected_epoch':best,'parameter_count':sum(p.numel() for p in final.parameters()),'reload_identical':True,'exact_missing_fallback_verified':True,'fallback_rows':int(fallback.sum()),'inner_dates':meta['inner_residual_dates'],'OOF_dates':meta['OOF_dates'],'plan_id':plan['plan_id'],'external_study_hash':external_hash(),'policy_source_hash':policy_source_hash(),'relative_bundle':str(dest.relative_to(T)),'reserved_access':False}
                                commit_bundle(dest,{'model.pt':path.read_bytes(),'predictions.npz':stream.getvalue(),'fit.json':fit,'inner_history.json':innerhistory,'refit_history.json':history},{'plan_id':plan['plan_id'],'external_study_hash':external_hash(),'reserved_access':False})
                            observed.observe(peak_vram_bytes=int(torch.cuda.max_memory_allocated()),outcome={'state':'durable_model_predictions_saved','relative_bundle':str(dest.relative_to(T)),'selected_epoch':best});del final,again;torch.cuda.empty_cache()
                        completed.append(fit);atomic_json(R/'reports/bar_fits_v1.json',{'completed_candidates':len(completed),'planned_candidates':6,'results':completed,'meter_records':meter.rows(),'reserved_access':False});progress('BAR_CANDIDATE_DURABLE',track=track,year=year,seed=seed,selected_epoch=best,completed_candidates=len(completed),planned_candidates=6,elapsed_seconds=meter.rows()[-1]['elapsed_seconds'])
                    except BaseException as error:
                        atomic_json(R/'reports'/('bar_failure_'+attempt_id.replace(':','_')+'.json'),{'type':type(error).__name__,'traceback':traceback.format_exc(),'attempt_id':attempt_id,'external_study_hash':external_hash(),'plan_id':plan['plan_id'],'meter_records':meter.rows(),'completed_results':completed,'reserved_access':False});raise
        progress('BAR_SIX_CANDIDATES_COMPLETED',completed_candidates=6,meter_seconds=sum(x['elapsed_seconds'] for x in meter.rows()))
    finally:meter.close()

def evaluate():
    from evaluation import analyse
    c=study();plan=load(PLAN);return analyse(R,T,c,plan,external_hash,intact,progress)
def validate():
    from evaluation import independently_validate
    c=study();plan=load(PLAN);return independently_validate(R,T,c,plan,external_hash,intact,progress)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','integration','register','prepare','run','evaluate','validate']);action=p.parse_args().action
    try:globals()[{'prepare':'prepare_data'}.get(action,action)]()
    except BaseException:print(traceback.format_exc(),flush=True);raise
