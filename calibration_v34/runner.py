"""Versioned genuine CUDA SIA prequential OOF; never changes frozen source."""
import os,sys,json,io,time,tempfile,traceback,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from signalforge.runtime import digest,file_hash,code_hash,atomic_json,commit_bundle,validate_bundle,now
from signalforge.panels import track_folds
from signalforge.v33_sia import prepare_sia,schema_from_specs
from signalforge.v33_corrections import mature_partitions
from signalforge.track_neural import ExplicitSourceCUDA
from policy_v3.v33_budget_free import ObservationMeter,device_lease,policy,policy_source_hash,admission,CooperativeCancellation
from build_v33_readiness import load_original,bytes_hash
from diagnostics_v3.audit import preserve,clean
from calibration_v34.api import Q,C,maturity,fit,apply,loss,monitor
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering');O=R/'reports/calibration_v34'
CONFIG=R/'calibration_v34/protocol_v1.json'
SUBJECTS=[('Main-A',2020,'I3'),('Nested-B',2022,'I4')]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def source_hash():return digest({str(p.relative_to(R)):file_hash(p) for p in sorted((R/'calibration_v34').glob('*')) if p.suffix in {'.py','.ps1'}})
def save(n,x):atomic_json(O/(n+'.json'),clean(x))
def event(stage,**kw):
    x={'created_at':now(),'stage':stage,'v34_code_hash':source_hash(),'reserved_access':False,'final_access':False,**kw};save('status',x);print(json.dumps(clean(x)),flush=True)
def immutable(path,x):
    if path.exists():assert load(path)==clean(x),'Immutable new receipt mismatch'
    else:atomic_json(path,clean(x))
def cfg():
    c=load(CONFIG);assert c['code_hash']==source_hash() and c['policy_hash']==policy_source_hash() and c['configuration_id']==digest({k:v for k,v in c.items() if k!='configuration_id'});return c
def tests():
    p=subprocess.run([sys.executable,'-m','pytest','calibration_v34/test_contract.py','-q','-p','no:cacheprovider'],capture_output=True,text=True);print(p.stdout+p.stderr,flush=True);save('tests',{'exit_code':p.returncode,'output':p.stdout+p.stderr,'code_hash':source_hash()});assert p.returncode==0

def register():
    preservation=preserve();c={'protocol_id':'sgqx-v34-SIA-genuine-mature-OOF-calibration-v1','code_hash':source_hash(),'policy_hash':policy_source_hash(),'frozen_v2_source_hash':code_hash(R),'subjects':SUBJECTS,'seeds':[11,37,71],'model':{'kind':'rgmf_gru','width':16,'lr':.001,'epochs':100,'optimizer':'AdamW default weight_decay .01','clip_gradient_norm':1.,'observed':'SIA raw observed + true indicator masks','elapsed':'zeros faithful frozen v2 adapter','source_mask':'explicit raw latest source values'},'OOF':{'minimum_past_distinct_dates':26,'block_dates':13,'inner_validation_dates':13,'minimum_inner_fit_distinct_dates':26,'expanding':True},'calibration':{'candidates':['identity','intercept','cqr'],'selection':'track-wide seed-average chronological inner normalized pinball; earliest identity tie','minimum_quantile_dates':[52,39,20,39,52],'expected_smaller_tail_date_floor':20,'unsupported':'exact identity fixed anchors','refit':'per-seed mature OOF before outer cutoff','ensembling':False},'uncertainty':{'blocks':[4,8,13],'draws':2000,'seed':20261008},'scope':'POST_RESULT_EXPLORATORY_USED_OUTER_YEARS; NOT INDEPENDENT CONFIRMATION','budget_free':True,'HPO':False,'BAR_training':False,'reserved_access':False,'final_access':False};c['configuration_id']=digest(c);immutable(CONFIG,c);save('preflight',preservation);event('PROTOCOL_FROZEN',configuration_id=c['configuration_id'])

def cohort(track,year,info):
    manifest,settings,panels,contexts,source=load_original(track);fs={i:track_folds(p,settings,track)[0][year] for i,p in panels.items()};common=set.intersection(*[set(zip(f['train'].loc[f['train'].context_eligible,'decision_time'],f['train'].loc[f['train'].context_eligible,'asset'])) for f in fs.values()]);tr=fs[info]['train'];tr=tr[[k in common for k in zip(tr.decision_time,tr.asset)]].sort_values(['decision_time','asset']);assert (pd.to_datetime(tr.max_dependency_available_at,utc=True)<=pd.to_datetime(tr.decision_time,utc=True)).all();spec=[s for s in manifest['features'] if s['source'] in settings['information_sets'][info]];schema=schema_from_specs(spec,panels[info].attrs['feature_names']);return tr,contexts[info],schema,panels[info].attrs['neural_contract'],source,fs[info]['outer_cutoff']

def block_input(tr,contexts,schema,contract,p):
    past=tr.iloc[p['train']];held=tr.iloc[p['oof']];cut=pd.Timestamp(p['cutoff']);maturity(past.assign(role='OOF'),cut);assert held.decision_time.min()==cut
    a=prepare_sia(past,held,contexts,cut,schema,neural_contract=contract)
    def mask(key):
        m=a[key]['observed'];return np.concatenate([m,np.ones_like(m),np.ones_like(m)],axis=-1)
    scale=held.asset.map(a['asset_scales']).to_numpy();arrays={'tx':a['train']['model_matrix'],'vx':a['test']['model_matrix'],'y':a['train_target'],'weights':a['weights'],'observed':mask('train'),'test_observed':mask('test'),'train_valid':a['source_valid'],'test_valid':a['test_source_valid'],'scale':scale}
    keys=lambda f:[[pd.Timestamp(d).isoformat(),str(s)] for d,s in zip(f.decision_time,f.asset)]
    meta={'schema_id':a['transform'].manifest()['manifest_id'],'normalizer_id':a['normalizer_id'],'target_id':bytes_hash(held.y),'fit_cutoff':cut.isoformat(),'training_decision_times':[pd.Timestamp(d).isoformat() for d in past.decision_time],'label_end':[pd.Timestamp(d).isoformat() for d in past.label_end],'label_available_at':[pd.Timestamp(d).isoformat() for d in past.label_available_at],'date_asset_keys':keys(held),'date_asset_keys_hash':digest(keys(held)),'assets':sorted(held.asset.unique()),'past_keys_hash':digest(keys(past)),'neural_args':a['neural_args'],'array_hashes':{k:bytes_hash(v) for k,v in arrays.items()},'past_dates':int(past.decision_time.nunique()),'predicted_dates':int(held.decision_time.nunique()),'max_source_dependency':str(past.max_dependency_available_at.max()),'transform':a['transform'].manifest()}
    return arrays,meta,held

def prepare_data():
    c=cfg();blocks=[]
    for track,year,info in SUBJECTS:
        tr,ctx,schema,contract,source,outercut=cohort(track,year,info);parts=mature_partitions(tr.decision_time,tr.label_end,tr.label_available_at,minimum=26,block=13)
        for i,p in enumerate(parts):
            a,m,held=block_input(tr,ctx,schema,contract,p);m.update(track=track,year=year,information=info,block=i+1,source_id=source['source_manifest_id'],outer_cutoff=outercut.isoformat());m['fold_id']=digest(m);blocks.append(m);event('OOF_INPUT_QUALIFIED',track=track,block=i+1,blocks=len(parts),past_dates=m['past_dates'],predicted_dates=m['predicted_dates'])
        assert sum(b['predicted_dates'] for b in blocks if b['track']==track)>=39
    immutable(O/'prepared.json',{'configuration_id':c['configuration_id'],'blocks':blocks,'planned_fits':len(blocks)*3});event('INPUT_QUALIFICATION_COMPLETE',blocks=len(blocks),planned_fits=len(blocks)*3)

class ObservedSIA(ExplicitSourceCUDA):
    def forward(self,*args,**kwargs):
        if (O/'STOP').exists() or (R/'.local/STOP').exists() or (R/'.local/PAUSE').exists():raise CooperativeCancellation('Explicit stop/pause')
        result=super().forward(*args,**kwargs)
        if self.model.training and getattr(self,'_training_source_valid',None) is not None:
            self.epoch=getattr(self,'epoch',0)+1
            if self.epoch%25==0:
                self.observation.observe(peak_vram_bytes=int(torch.cuda.max_memory_allocated()),outcome={'epochs_observed':self.epoch});event('GPU_OOF_EPOCH',**self.live,epoch=self.epoch)
        return result

def train_fit(a,m,seed,meter,phase='study',epochs=100):
    c=cfg();identity=digest({'configuration_id':c['configuration_id'],'fold_id':m['fold_id'],'seed':seed,'phase':phase,'epochs':epochs});dest=T/'artifacts/sia_oof_v34'/phase/identity
    if (dest/'receipt.json').exists():validate_bundle(dest);return load(dest/'fit.json')
    previous=[r for r in meter.rows() if r['root_identity']==identity]
    retry_of=None;attempt_id=identity
    if previous:
        parent=previous[-1]
        if parent['state'] not in ['FAILED','INTERRUPTED']:raise PermissionError('Unresolved or committed meter identity')
        retry_of=parent['identity'];attempt_id=digest({'root':identity,'retry':len(previous)})
    checkpoint=T/'checkpoints/sia_oof_v34'/identity/'checkpoint.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    event('GPU_OOF_FIT_START',track=m['track'],block=m['block'],seed=seed,phase=phase,identity=identity)
    try:
        with meter.attempt(attempt_id,metadata={'configuration_id':c['configuration_id'],'fold_id':m['fold_id'],'phase':phase,'seed':seed},retry_of=retry_of) as observation:
            torch.cuda.reset_peak_memory_stats();model=ObservedSIA(kind='rgmf_gru',width=16,lr=.001,epochs=epochs,seed=seed,**m['neural_args']);model.observation=observation;model.live={'track':m['track'],'block':m['block'],'seed':seed};model.fit(a['tx'],a['y'],1.,weights=a['weights'],observed=a['observed'],elapsed=np.zeros_like(a['tx']),source_valid=a['train_valid'],checkpoint_path=checkpoint,checkpoint_seconds=15)
            pred_args={'observed':a['test_observed'],'source_valid':a['test_valid']};mean,q=model.predict(a['vx'],**pred_args);q=q*a['scale'][:,None];assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
            with tempfile.TemporaryDirectory(dir=T/'tmp') as tmp:
                path=Path(tmp)/'model.pt';model.save(path);reload=ExplicitSourceCUDA.load(path);rq=reload.predict(a['vx'],**pred_args)[1]*a['scale'][:,None];assert np.array_equal(q,rq),'Actual CUDA reload differs'
                frame=pd.DataFrame(m['date_asset_keys'],columns=['decision_time','asset']);frame['seed']=seed;frame[C]=q;frame['scale']=a['scale'];stream=io.BytesIO();np.savez_compressed(stream,quantiles=q,scale=a['scale'])
                ck=torch.load(checkpoint,map_location='cpu',weights_only=False);receipt={'identity':identity,'attempt_id':attempt_id,'track':m['track'],'year':m['year'],'block':m['block'],'seed':seed,'phase':phase,'relative_bundle':str(dest.relative_to(T)),'fold_id':m['fold_id'],'configuration_id':c['configuration_id'],'reload_bitwise_equal':True,'checkpoint_hash':file_hash(checkpoint),'checkpoint_path':str(checkpoint),'epochs':epochs,'prediction_hash':bytes_hash(q),'rows':len(q),'peak_vram_bytes':int(torch.cuda.max_memory_allocated()),'device':torch.cuda.get_device_name(),'input_metadata':m}
                commit_bundle(dest,{'model.pt':path.read_bytes(),'predictions.npz':stream.getvalue(),'predictions.csv':frame.to_csv(index=False).encode(),'fit.json':receipt},{'configuration_id':c['configuration_id'],'reserved_access':False});observation.observe(peak_vram_bytes=receipt['peak_vram_bytes'],outcome={'reload_bitwise_equal':True,'bundle':receipt['relative_bundle']})
            del model,reload;torch.cuda.empty_cache()
        event('GPU_OOF_FIT_SAVED',track=m['track'],block=m['block'],seed=seed,rows=len(q),phase=phase);return receipt
    except BaseException as error:
        save('failure_'+attempt_id,{'identity':identity,'attempt_id':attempt_id,'traceback':traceback.format_exc(),'error_type':type(error).__name__,'checkpoint':str(checkpoint),'configuration_id':c['configuration_id']});raise

def integration():
    c=cfg();prepared=load(O/'prepared.json');meter=ObservationMeter(T/'ledger/v33_budget_free_v3_sia_oof_v34_integration.sqlite');results=[]
    try:
        with device_lease(2*1024**3):
            for track,year,info in SUBJECTS:
                tr,ctx,schema,contract,source,cut=cohort(track,year,info);p=mature_partitions(tr.decision_time,tr.label_end,tr.label_available_at,minimum=26,block=13)[0];a,m,h=block_input(tr,ctx,schema,contract,p);m=next(b for b in prepared['blocks'] if b['track']==track and b['block']==1);results.append(train_fit(a,m,11,meter,'integration',2))
            identity=digest({'configuration_id':c['configuration_id'],'fixture':'intentional_failure'})
            if not any(r['identity']==identity for r in meter.rows()):
                try:
                    with meter.attempt(identity,metadata={'fixture':True}):raise RuntimeError('Intentional retained failure fixture')
                except RuntimeError:pass
            try:
                with meter.attempt(results[0]['attempt_id']):pass
            except ValueError:pass
            else:raise AssertionError('Duplicate identity permitted')
        save('integration',{'passed':True,'results':results,'meter':meter.rows(),'policy_hash':policy_source_hash(),'configuration_id':c['configuration_id'],'failure_fixture_retained':True,'duplicate_rejected':True,'no_budget':True})
    finally:meter.close()
    blocks=prepared['blocks'];fits=[{k:b[k] for k in ['fold_id','source_id','schema_id','normalizer_id','target_id','date_asset_keys_hash']}|{'seed':seed} for b in blocks for seed in [11,37,71]]
    q={'source_tree_hash':code_hash(R),'policy_source_hash':policy_source_hash(),'passed':True,'cuda_inputs_validated':True,'plan_contract_id':c['configuration_id'],'folds':{b['fold_id']:b for b in blocks},'validated_fits':fits,'integration_receipt_hash':file_hash(O/'integration.json')};q['qualification_id']=digest(q)
    plan={'policy_id':policy()['policy_id'],'policy_source_hash':policy_source_hash(),'policy_hash':digest(policy()),'source_tree_hash':code_hash(R),'local_only':True,'reserved_access':False,'final_access':False,'plan_contract_id':c['configuration_id'],'qualification_id':q['qualification_id'],'fits':fits};plan['plan_id']=digest(plan);immutable(O/'qualification.json',q);immutable(O/'plan.json',plan);save('admission',admission(plan,q,experiment_requested=True));event('REAL_CUDA_INTEGRATION_PASS',checks=2,planned_fits=len(fits))

def run():
    c=cfg();preserve();plan=load(O/'plan.json');qualification=load(O/'qualification.json');admission(plan,qualification,experiment_requested=True);prepared=load(O/'prepared.json');assert prepared['configuration_id']==c['configuration_id'];results=[];meter=ObservationMeter(T/'ledger/v33_budget_free_v3_sia_oof_v34_study.sqlite')
    try:
        with device_lease(2*1024**3):
            for track,year,info in SUBJECTS:
                tr,ctx,schema,contract,source,outercut=cohort(track,year,info);parts=mature_partitions(tr.decision_time,tr.label_end,tr.label_available_at,minimum=26,block=13)
                for i,p in enumerate(parts):
                    a,m,h=block_input(tr,ctx,schema,contract,p);expected=next(b for b in prepared['blocks'] if b['track']==track and b['block']==i+1)
                    for k in m:assert clean(m[k])==expected[k],k
                    for seed in [11,37,71]:
                        results.append(train_fit(a,expected,seed,meter));save('fits',{'results':results,'planned':prepared['planned_fits'],'configuration_id':c['configuration_id']});event('OOF_DURABLE_PROGRESS',completed=len(results),planned=prepared['planned_fits'],track=track,block=i+1,seed=seed,measured_seconds=sum(r['elapsed_seconds'] for r in meter.rows()))
                    del a
        assert len(results)==prepared['planned_fits'];save('oof_completion',{'configuration_id':c['configuration_id'],'results':results,'meter_records':meter.rows(),'successful_fits':len(results),'preservation':preserve()});event('GENUINE_SIA_OOF_COMPLETE',fits=len(results))
    finally:meter.close()

def calibrate():
    c=cfg();completion=load(O/'oof_completion.json');rows=[];prepared=load(O/'prepared.json')
    for track,year,info in SUBJECTS:
        tr,ctx,schema,contract,source,cut=cohort(track,year,info);bykey=tr.set_index([tr.decision_time.astype(str),tr.asset])
        for result in completion['results']:
            if result['track']!=track:continue
            b=T/result['relative_bundle'];validate_bundle(b);f=pd.read_csv(b/'predictions.csv');f['track']=track;f['year']=year;f['role']='OOF';f['decision_time']=pd.to_datetime(f.decision_time,utc=True)
            ix=pd.MultiIndex.from_arrays([f.decision_time.astype(str),f.asset]);targets=bykey.loc[ix];f['target']=targets.y.to_numpy();f['label_end']=targets.label_end.to_numpy();f['label_available_at']=targets.label_available_at.to_numpy();rows.append(f)
    allf=pd.concat(rows,ignore_index=True);allf.to_csv(O/'genuine_sia_oof.csv',index=False);parameters={};thresholds={};selection=[]
    for track,year,info in SUBJECTS:
        f=allf[allf.track==track].copy();dates=sorted(f.decision_time.unique());innercut=pd.Timestamp(dates[-13]);outercut=pd.Timestamp(next(b for b in prepared['blocks'] if b['track']==track)['outer_cutoff']);inner=f[(f.decision_time<innercut)&(pd.to_datetime(f.label_end,utc=True)<=innercut)&(pd.to_datetime(f.label_available_at,utc=True)<=innercut)];val=f[f.decision_time.isin(dates[-13:])];assert inner.decision_time.nunique()>=26 and val.decision_time.nunique()==13
        scores={}
        for kind in ['identity','intercept','cqr']:
            scores[kind]=np.mean([loss(val[val.seed==seed],apply(val[val.seed==seed][C],val[val.seed==seed].scale,fit(inner[inner.seed==seed],kind,innercut))) for seed in [11,37,71]])
        chosen=min(scores,key=scores.get);selection.append({'track':track,'inner_dates':inner.decision_time.nunique(),'validation_dates':13,'scores':scores,'selected':chosen,'outer_used':False});parameters[track]={str(seed):fit(f[f.seed==seed],chosen,outercut) for seed in [11,37,71]};maturity(f,outercut)
        width=(f.q95-f.q05)/f.scale;error=np.abs(f.target-f.q50)/f.scale;thresholds[track]={'width90_p95':float(width.quantile(.95)),'width90_p99':float(width.quantile(.99)),'error_p95':float(error.quantile(.95)),'error_p99':float(error.quantile(.99)),'distinct_dates':len(dates),'outer_used':False,'operationally_frozen':False}
    receipt={'configuration_id':c['configuration_id'],'parameters':parameters,'selection':selection,'thresholds':thresholds,'OOF_hash':file_hash(O/'genuine_sia_oof.csv'),'outer_outcomes_read_during_fit':False,'candidate_ensembles':False};receipt['parameter_id']=digest(receipt);immutable(O/'calibration_frozen.json',receipt);event('TRAIN_ONLY_CALIBRATION_FROZEN',rows=len(allf),parameter_id=receipt['parameter_id'],selection=selection)

def evaluate():
    from calibration_v34.evaluation import evaluate as execute
    execute()
def validate():
    from calibration_v34.evaluation import validate as execute
    execute()
if __name__=='__main__':globals()[sys.argv[1]]()