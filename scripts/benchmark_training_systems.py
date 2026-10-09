"""Real development E10 replicas, excluded from HPO, selection and final evidence."""
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import copy,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from signalforge.runtime import paths,atomic_json,digest,validate_bundle,commit_bundle,now,ensure_gpu_owner
from signalforge.resources import research_gpu_stage
from signalforge.neural import normalized_loss
from signalforge.models import NeuralCUDA,QUANTILES
from signalforge.features import TrainTransform
from signalforge.metrics import quantile_loss


def worker_threads(_):
    torch.set_num_threads(1)


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
    return result,time.perf_counter()-start


def main():
    torch.set_num_threads(4)
    repo,runtime=paths()
    document=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
    if document.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: completed development required')
    row=next(r for r in document['results'] if r['family']=='mlp' and r['outer_year']==2022 and r['seed']==11)
    directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
    manifest=json.loads((directory/'transform.json').read_text())
    from signalforge.data import auxiliary_rows
    from signalforge.auxiliary import sequences
    from signalforge.panels import track_folds
    from signalforge.development import preprocess
    acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
    frame,raw=sequences(auxiliary_rows(acquisition['records']))
    study=json.loads((repo/'configs/study.json').read_text());folds,blocked=track_folds(frame,study,'Auxiliary-C')
    assert not blocked
    fold=folds[2022];train,test=fold['train'],fold['test']
    cut=pd.Timestamp('2022-01-01T00:00Z')-pd.Timedelta(nanoseconds=1)
    tx,ex,scale,normalizer,transform=preprocess(raw,train,test,cut)
    assert normalizer==row['normalizer_id'] and float(scale)==row['scale']
    assert np.array_equal(transform.center,np.asarray(manifest['center'])) and np.array_equal(transform.scale,np.asarray(manifest['scale']))
    x=tx.reshape(len(tx),-1).astype(np.float32);y=train.y.to_numpy(dtype=np.float32)
    ex=ex.reshape(len(ex),-1).astype(np.float32);ey=test.y.to_numpy(dtype=np.float32)
    protocol={'representative':{'family':'mlp','outer_year':2022,'seed':11},'model_receipt_id':digest(receipt),
        'source_data_id':row['data_id'],'normalizer_id':normalizer,'n_mature_training_dates':train.decision_time.nunique(),
        'n_outer_dates':test.decision_time.nunique(),'epochs':100,'counterbalance':['FP32','BF16','BF16','FP32']*3,
        'effective_batch':'all mature training dates','microbatches':[32,128],
        'FP32_equivalence_tolerance':1e-4,'BF16_relative_loss_tolerance':.05,
        'BF16_max_normalized_abs_delta':.05,
        'workers':[0,2,4],'replicas_are_scientific_candidates':False,'adoption_allowed':False,'reserved_access':False}
    atomic_json(repo/'reports/real_training_systems_protocol.json',protocol)
    with research_gpu_stage(repo,runtime,'real_training_systems',900) as admission:
        saved=NeuralCUDA.load(directory/'model.bin')
        original=json.loads((directory/'predictions.json').read_text())
        torch.manual_seed(11);torch.cuda.manual_seed_all(11)
        initial=saved.build(x.shape[1]).cuda().state_dict()
        gx=torch.tensor(x,device='cuda');gy=torch.tensor(y,device='cuda');ge=torch.tensor(ex,device='cuda')
        reloaded=saved.predict(ex)
        reload_delta=max(float(np.max(np.abs(a-np.asarray(b)))) for a,b in zip(reloaded,[original['mean'],original['quantiles']]))
        if reload_delta>2e-6:raise ValueError('Real training benchmark input/model reload mismatch')
        def fresh():
            model=saved.build(x.shape[1]).cuda();model.load_state_dict(initial)
            return model,torch.optim.AdamW(model.parameters(),lr=saved.config['lr'])
        def train_model(mode='FP32',batch=None,compile_model=False,recovery=False,overlap=False,serial_preprocessing=False):
            ensure_gpu_owner()
            model,optimizer=fresh();cold=0.;compiled=model
            if compile_model:
                compiled=torch.compile(model)
                _,cold=timed(lambda:compiled(gx))
            pool=ThreadPoolExecutor(max_workers=1) if overlap else None
            pending=None
            start=time.perf_counter()
            for epoch in range(100):
                if serial_preprocessing and epoch%10==0:
                    rebuilt=transform.transform(raw[train.sequence_index].reshape(-1,raw.shape[-1])).reshape(len(x),-1).astype(np.float32)
                    assert np.array_equal(rebuilt,x)
                if pool and epoch%10==0:
                    if pending is not None:assert np.array_equal(pending.result(),x)
                    pending=pool.submit(lambda:transform.transform(raw[train.sequence_index].reshape(-1,raw.shape[-1])).reshape(len(x),-1).astype(np.float32))
                optimizer.zero_grad(set_to_none=True)
                chunk=len(x) if batch is None else batch
                for offset in range(0,len(x),chunk):
                    xx=gx[offset:offset+chunk];yy=gy[offset:offset+chunk]
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='BF16'):
                        mean,q=compiled(xx);loss=normalized_loss(yy,mean,q,scale)*len(xx)/len(x)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite real benchmark training loss')
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if recovery and epoch==49:
                    # Exact optimizer boundary with actual disk round trip; no synthetic data.
                    path=runtime/'tmp'/('systems-recovery-'+digest({'pid':os.getpid(),'time':now()})+'.pt')
                    path.parent.mkdir(exist_ok=True)
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':50},path)
                    state=torch.load(path,weights_only=False,map_location='cuda')
                    model,optimizer=fresh();model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);compiled=model
                    path.unlink()
            if pool:
                assert np.array_equal(pending.result(),x);pool.shutdown()
            torch.cuda.synchronize();seconds=time.perf_counter()-start+cold
            ensure_gpu_owner()
            model.eval()
            with torch.no_grad():outputs=[v.float().cpu().numpy() for v in model(ge)]
            loss=float(quantile_loss(ey,outputs[1],QUANTILES,scale).mean())
            return {'seconds_including_cold':seconds,'cold_seconds':cold,'pinball':loss,
                'useful_examples':len(x)*100,'distinct_training_dates':len(x),'outputs':outputs}
        reference=train_model();probes={};precision=[]
        for mode in protocol['counterbalance']:
            result=train_model(mode=mode)
            raw_delta=max(float(np.max(np.abs(a-b))) for a,b in zip(reference['outputs'],result.pop('outputs')))
            delta=raw_delta/scale
            relative_loss_delta=abs(result['pinball']-reference['pinball'])/max(reference['pinball'],1e-8)
            passed=raw_delta<=protocol['FP32_equivalence_tolerance'] if mode=='FP32' else delta<=.05 and relative_loss_delta<=.05
            result.update(mode=mode,max_abs_delta=raw_delta,max_normalized_abs_delta=delta,
                relative_pinball_delta=relative_loss_delta,quality_passed=passed)
            precision.append(result)
        probes['training_precision']=precision
        probes['original_model_reload']={'passed':True,'max_abs_delta':reload_delta}
        for name,kwargs in [('microbatch32',{'batch':32}),('microbatch128',{'batch':128}),
                            ('compile',{'compile_model':True}),('recovery',{'recovery':True}),
                            ('CPU_serial',{'serial_preprocessing':True}),('CPU_overlap',{'overlap':True})]:
            try:
                result=train_model(**kwargs)
                delta=max(float(np.max(np.abs(a-b))) for a,b in zip(reference['outputs'],result.pop('outputs')))
                result.update(max_abs_delta=delta,passed=delta<=1e-4,state='COMPLETED_DIAGNOSTIC',adopted=False)
                probes[name]=result
            except Exception as error:
                if str(error).startswith(('BLOCKED_GPU','BLOCKED_RESOURCE','PAUSED_BUDGET')):raise
                probes[name]={'state':'FAILED_BENCHMARK','error_type':type(error).__name__,'reason':str(error),'adopted':False}
        loaders=[]
        for workers in protocol['workers']:
            ensure_gpu_owner()
            try:
                start=time.perf_counter();parts=[]
                loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),
                    batch_size=128,shuffle=False,num_workers=workers,pin_memory=True,worker_init_fn=worker_threads,
                    multiprocessing_context='spawn' if workers else None)
                for xx,yy in loader:
                    parts.append((xx.to('cuda',non_blocking=True),yy.to('cuda',non_blocking=True)))
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                xx=torch.cat([p[0] for p in parts]).cpu().numpy();yy=torch.cat([p[1] for p in parts]).cpu().numpy()
                loaders.append({'workers':workers,'seconds_including_startup_and_copy':seconds,'useful_examples':len(x),
                    'passed':np.array_equal(xx,x) and np.array_equal(yy,y),'adopted':False})
            except Exception as error:loaders.append({'workers':workers,'state':'FAILED_BENCHMARK','reason':str(error),'adopted':False})
        probes['dataloader_workers_and_pinned_copy']=loaders
        rawtrain=raw[train.sequence_index]
        start=time.perf_counter()
        for _ in range(30):uncached=transform.transform(rawtrain.reshape(-1,raw.shape[-1])).reshape(len(x),-1).astype(np.float32)
        uncached_seconds=time.perf_counter()-start
        start=time.perf_counter()
        for _ in range(30):cached=x.copy()
        cached_seconds=time.perf_counter()-start
        probes['cached_preprocessing']={'uncached_30_seconds':uncached_seconds,'cached_30_seconds':cached_seconds,
            'passed':np.array_equal(uncached,cached),'transform_refitted':False,'adopted':False}
        reference.pop('outputs')
        result={'state':'COMPLETED_REAL_TRAINING_SYSTEMS_DIAGNOSTIC','protocol':protocol,'reference':reference,'probes':probes,
            'actual_CUDA_used':True,'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'resource_admission':admission,'created_at':now(),'qualified_research_systems':False,'reserved_access':False,
            'limitations':['one fixed actual MLP workload','replicas excluded from HPO and scientific candidates','no optimization adopted','recovery is an optimizer-state disk roundtrip, not an OS shutdown test']}
    identity=digest(result);commit_bundle(runtime/'artifacts/real_training_systems'/identity,{'results.json':result},{'model_receipt_id':digest(receipt),'source_data_id':row['data_id']})
    result['artifact_id']=identity;atomic_json(repo/'reports/real_training_systems_qualification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
