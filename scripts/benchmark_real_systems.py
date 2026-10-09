"""Counterbalanced E10 inference on one actual frozen development MLP bundle.

Repeated passes measure systems throughput, never independent research samples.
No weights, seeds, hyperparameters or predictions used for model selection change.
"""
import json,time,copy,os
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)
from signalforge.runtime import paths,validate_bundle,digest,atomic_json,commit_bundle,now,file_hash
from signalforge.resources import research_gpu_stage
from signalforge.models import NeuralCUDA,QUANTILES
from signalforge.features import TrainTransform
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences
from signalforge.metrics import quantile_loss
repo,runtime=paths()
document=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
if document.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: completed development required')
# Outcome-independent representative: last registered fold, MLP, first fixed seed.
row=next(r for r in document['results'] if r['family']=='mlp' and r['outer_year']==2022 and r['seed']==11)
directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
saved=json.loads((directory/'predictions.json').read_text());manifest=json.loads((directory/'transform.json').read_text())
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text());frame,raw=sequences(auxiliary_rows(acquisition['records']))
index={pd.Timestamp(d).isoformat():i for i,d in enumerate(frame.decision_time)}
raw=raw[[index[pd.Timestamp(d).isoformat()] for d in saved['decision_time']]]
transform=TrainTransform();transform.center=np.asarray(manifest['center']);transform.scale=np.asarray(manifest['scale'])
x=transform.transform(raw.reshape(-1,raw.shape[-1])).reshape(len(raw),-1).astype(np.float32)
protocol={'representative':{'family':'mlp','outer_year':2022,'seed':11},'model_receipt_id':digest(receipt),
    'source_data_id':row['data_id'],'n_distinct_dates':len(raw),'repeated_passes':30,'counterbalance':['FP32','BF16','BF16','FP32']*3,
    'bf16_max_normalized_abs_delta':.03,'bf16_max_relative_pinball_delta':.02,'fp32_compile_max_abs_delta':1e-5,
    'batch_max_abs_delta':1e-5,'adoption_allowed':False,'repetition_is_new_evidence':False,'reserved_access':False}
protocol['statistics_probe']={'left':'ridge','right':'lightgbm','block_weeks':8,'draws':2000,'dtype':'float64','max_abs_tolerance':1e-12,'actual_development_predictions_only':True}
atomic_json(repo/'reports/real_systems_protocol.json',protocol)
def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,time.perf_counter()-start
with research_gpu_stage(repo,runtime,'real_development_systems',900):
    model=NeuralCUDA.load(directory/'model.bin').model.eval();tx=torch.tensor(x,device='cuda')
    def predict(model=model,bf16=False,inputs=tx):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            return model(inputs)
    reference,_=timed(predict)
    reference=[v.float().cpu().numpy() for v in reference]
    delta=max(float(np.max(np.abs(a-np.asarray(b)))) for a,b in zip(reference,[saved['mean'],saved['quantiles']]))
    if delta>2e-6:raise ValueError('Real systems input/model reload mismatch')
    baseline=float(quantile_loss(np.asarray(saved['y']),reference[1],QUANTILES,row['scale']).mean())
    timings=[];last={}
    for mode in protocol['counterbalance']:
        outputs,seconds=timed(lambda:[predict(bf16=mode=='BF16') for _ in range(30)])
        last[mode]=[v.float().cpu().numpy() for v in outputs[-1]]
        timings.append({'mode':mode,'seconds':seconds,'useful_processed_examples':len(x)*30,'distinct_dates':len(x)})
    bfdelta=max(float(np.max(np.abs(a-b))) for a,b in zip(reference,last['BF16']))/row['scale']
    bfloss=float(quantile_loss(np.asarray(saved['y']),last['BF16'][1],QUANTILES,row['scale']).mean())
    quality=bfdelta<=.03 and abs(bfloss-baseline)<=.02*max(baseline,1e-8)
    probes={'precision':{'timings':timings,'max_normalized_abs_delta':bfdelta,'FP32_pinball':baseline,'BF16_pinball':bfloss,'quality_passed':quality,'adopted':False},
        'reload':{'max_abs_delta':delta,'passed':True}}
    batches=[]
    for size in [1,8,32,len(x)]:
        chunks,seconds=timed(lambda:[predict(inputs=t) for t in tx.split(size)])
        outputs=[torch.cat([v[i] for v in chunks]).float().cpu().numpy() for i in [0,1]]
        delta=max(float(np.max(np.abs(a-b))) for a,b in zip(reference,outputs))
        batches.append({'batch':size,'seconds':seconds,'useful_examples':len(x),'max_abs_delta':delta,'passed':delta<1e-5})
    probes['batching']=batches
    try:
        compiled=torch.compile(copy.deepcopy(model))
        cold,cold_time=timed(lambda:predict(model=compiled))
        warm,warm_time=timed(lambda:[predict(model=compiled) for _ in range(30)])
        delta=max(float(np.max(np.abs(a-b.float().cpu().numpy()))) for a,b in zip(reference,cold))
        probes['compile']={'cold_seconds':cold_time,'warm_30_seconds':warm_time,'max_abs_delta':delta,'passed':delta<1e-5,'adopted':False,
            'cold_cost_included':True,'useful_examples':len(x)*31}
    except Exception as error:
        probes['compile']={'state':'FAILED_BENCHMARK','error_type':type(error).__name__,'reason':str(error),'adopted':False}
    from signalforge.statistics import calendar_indices
    analysis_path=repo/'reports/auxiliary_development_analysis.json'
    analysis=json.loads(analysis_path.read_text())
    left=analysis['forecast_rows']['ridge'];right=analysis['forecast_rows']['lightgbm']
    if [r['decision_time'] for r in left]!=[r['decision_time'] for r in right] or len(left)!=260:
        raise ValueError('Real statistics date grid mismatch')
    if any(a['y']!=b['y'] or a['normalizer_id']!=b['normalizer_id'] or a['scale']!=b['scale'] for a,b in zip(left,right)):
        raise ValueError('Real statistics target/scale mismatch')
    dates=[r['decision_time'] for r in left];indices,metadata=calendar_indices(dates,8,2000)
    y=np.asarray([r['y'] for r in left]);scales=np.asarray([r['scale'] for r in left])
    delta=quantile_loss(y,np.asarray([r['quantiles'] for r in left]),QUANTILES,scales)-quantile_loss(y,np.asarray([r['quantiles'] for r in right]),QUANTILES,scales)
    start=time.perf_counter();cpu=delta[indices].mean(1);cpu_seconds=time.perf_counter()-start
    def cuda_statistics():
        values=torch.tensor(delta,dtype=torch.float64,device='cuda');draws=torch.tensor(indices,dtype=torch.long,device='cuda')
        return values[draws].mean(1)
    gpu,gpu_seconds=timed(cuda_statistics)
    max_delta=float(np.max(np.abs(cpu-gpu.cpu().numpy())))
    probes['real_FP64_statistics']={'source_analysis_sha256':file_hash(analysis_path),'n_distinct_dates':260,'draws':2000,
        'shared_index_id':metadata['index_id'],'CPU_seconds':cpu_seconds,'CUDA_seconds_including_transfer':gpu_seconds,
        'max_abs_delta':max_delta,'passed':max_delta<=1e-12,'adopted':False,'independent_sample_count':260}
    result={'state':'COMPLETED_REAL_DEVELOPMENT_SYSTEMS_DIAGNOSTIC','protocol':protocol,'probes':probes,'actual_CUDA_used':True,
        'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'created_at':now(),
        'qualified_research_systems':False,'limitations':['one fixed actual MLP workload','inference timing only; training optimization matrix incomplete'],
        'reserved_access':False,'no_model_or_scientific_selection':True}
atomic_json(repo/'reports/real_systems_qualification.json',result)
identity=digest(result);commit_bundle(runtime/'artifacts/real_systems'/identity,{'results.json':result},{'model_receipt_id':digest(receipt),'source_data_id':row['data_id']})
print(json.dumps(result,indent=2))
import os,sys
# A child would see this process's live CUDA context as a foreign owner.
# Process replacement closes that context and retains the stage time bound.
sys.stdout.flush();sys.stderr.flush()
os.execv(sys.executable,[sys.executable,'scripts/benchmark_training_systems.py'])
