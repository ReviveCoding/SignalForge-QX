"""Development-only synthetic systems probes on the actual single CUDA GPU."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from signalforge.neural import Encoder,normalized_loss,require_cuda
from signalforge.runtime import paths,atomic_json,gpu_lease,file_hash,now
from signalforge.features import TrainTransform


def worker_init(_):
    torch.set_num_threads(1)


def timed(fn):
    torch.cuda.synchronize()
    start=time.perf_counter()
    result=fn()
    torch.cuda.synchronize()
    return result,time.perf_counter()-start


def main():
    repo,runtime=paths()
    with gpu_lease(runtime):
        require_cuda();torch.set_num_threads(4);torch.manual_seed(11)
        device=torch.cuda.get_device_name(0);free,total=torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(.75,(free-1024**3)/total))
        model=Encoder(12,64,'mlp').cuda()
        x=torch.randn(512,12,device='cuda');y=.2*x[:,0]-.1*x[:,1]
        cpu=copy.deepcopy(model).cpu()
        with torch.no_grad():
            cp,_=cpu(x.cpu());gp,_=model(x)
        probes={'S1':{'max_prediction_abs_delta':float((gp.cpu()-cp).abs().max()),
                      'passed':bool(torch.allclose(gp.cpu(),cp,atol=1e-5,rtol=1e-5))}}
        def inference(m=model,bf=False):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf):
                return m(x)
        fp,t_fp=timed(lambda:[inference() for _ in range(100)])
        bf,t_bf=timed(lambda:[inference(bf=True) for _ in range(100)])
        delta=max(float((a-b.float()).abs().max()) for a,b in zip(fp[-1],bf[-1]))
        probes['S2']={'fp32_seconds':t_fp,'bf16_seconds':t_bf,'useful_examples':51200,
                      'max_abs_delta':delta,'passed':delta<.03,'adopted':delta<.03 and t_bf<t_fp}
        try:
            compiled=torch.compile(model)
            cold,cold_seconds=timed(lambda:inference(compiled))
            warm,warm_seconds=timed(lambda:[inference(compiled) for _ in range(100)])
            delta=max(float((a-b).abs().max()) for a,b in zip(fp[-1],cold))
            from torch._dynamo.utils import counters
            probes['S3']={'cold_seconds':cold_seconds,'warm_100_seconds':warm_seconds,'eager_100_seconds':t_fp,
                          'max_abs_delta':delta,'passed':delta<1e-5,'adopted':delta<1e-5 and cold_seconds+warm_seconds<t_fp,
                          'dynamo_counters':{str(k):{str(a):int(b) for a,b in v.items()} for k,v in counters.items()}}
        except Exception as e:
            probes['S3']={'passed':False,'error_type':type(e).__name__,'adopted':False}
        full=copy.deepcopy(model);micro=copy.deepcopy(model)
        full.zero_grad();normalized_loss(y,*full(x),1).backward()
        micro.zero_grad()
        for a in range(0,len(y),64):
            (normalized_loss(y[a:a+64],*micro(x[a:a+64]),1)*64/len(y)).backward()
        gradient_delta=max(float((a.grad-b.grad).abs().max()) for a,b in zip(full.parameters(),micro.parameters()))
        probes['S4']={'effective_batch':512,'microbatch':64,'optimizer_steps':1,'max_gradient_abs_delta':gradient_delta,'passed':gradient_delta<1e-5}
        loader_results=[]
        dataset=TensorDataset(x.cpu(),y.cpu())
        for workers in [0,2,4]:
            kwargs={'multiprocessing_context':'spawn','worker_init_fn':worker_init} if workers else {}
            loader=DataLoader(dataset,batch_size=64,num_workers=workers,pin_memory=True,shuffle=False,**kwargs)
            start=time.perf_counter();seen=[]
            for a,b in loader:
                seen.append(a)
                model(a.cuda(non_blocking=True))
            torch.cuda.synchronize()
            loader_results.append({'workers':workers,'seconds':time.perf_counter()-start,'useful_examples':len(dataset),
                                   'passed':torch.equal(torch.cat(seen),x.cpu())})
        probes['S5']=loader_results
        def prep():
            a=np.arange(512*12,dtype=float).reshape(512,12)
            return np.sin(a).sum()
        _,serial=timed(lambda:[(prep(),inference()) for _ in range(30)])
        with ThreadPoolExecutor(max_workers=1) as pool:
            def overlap():
                for _ in range(30):
                    f=pool.submit(prep);inference();f.result()
            _,parallel=timed(overlap)
        probes['S6']={'serial_seconds':serial,'overlap_seconds':parallel,'passed':True,
                      'adopted':False,'cpu_workers':1,'same_deterministic_input':True,
                      'decision_reason':'One noisy timing pair is insufficient for operational adoption'}
        a=x.cpu().numpy();dates=['2020-01-01']*len(a)
        transform=TrainTransform().fit(a,dates,'2020-01-02T00:00Z')
        t=time.perf_counter();features=transform.transform(a);uncached=time.perf_counter()-t
        path=runtime/'cache/systems_features.npy';np.save(path,features)
        t=time.perf_counter();loaded=np.load(path);cached=time.perf_counter()-t
        probes['S7']={'uncached_seconds':uncached,'cached_seconds':cached,'passed':bool(np.array_equal(features,loaded)),
                      'fit_cutoff':transform.fit_cutoff,'cache_sha256':file_hash(path)}
        def step(m,opt):
            opt.zero_grad();loss=normalized_loss(y,*m(x),1);loss.backward();opt.step()
        uninterrupted=copy.deepcopy(model);opt1=torch.optim.AdamW(uninterrupted.parameters(),lr=.001)
        for _ in range(16):step(uninterrupted,opt1)
        interrupted=copy.deepcopy(model);opt2=torch.optim.AdamW(interrupted.parameters(),lr=.001)
        for _ in range(8):step(interrupted,opt2)
        checkpoint=runtime/'artifacts/systems_checkpoint.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        tmp=checkpoint.with_suffix('.partial')
        with tmp.open('wb') as f:
            torch.save({'model':interrupted.state_dict(),'optimizer':opt2.state_dict(),'step':8,'sample_cursor':4096,
                        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'data_hash':file_hash(path)},f)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,checkpoint)
        saved=torch.load(checkpoint,weights_only=True)
        resumed=copy.deepcopy(model);resumed.load_state_dict(saved['model'])
        opt3=torch.optim.AdamW(resumed.parameters(),lr=.001);opt3.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        for _ in range(8):step(resumed,opt3)
        delta=max(float((a-b).abs().max()) for a,b in zip(uninterrupted.parameters(),resumed.parameters()))
        probes['S8']={'checkpoint_sha256':file_hash(checkpoint),'boundary_step':8,'resumed_final_step':16,
                      'max_parameter_abs_delta':delta,'passed':delta==0,'scope':'checkpoint/reload interruption simulation; process-kill recovery not yet qualified'}
        probes['S9']=[]
        model.eval()
        with torch.no_grad():
            reference=model(x)
            for batch in [1,8,64]:
                for _ in range(5):model(x[:batch])
                times=[]
                for _ in range(30):
                    _,elapsed=timed(lambda:model(x[:batch]));times.append(elapsed)
                mean=torch.cat([model(z)[0] for z in x.split(batch)])
                delta=float((mean-reference[0]).abs().max())
                probes['S9'].append({'batch':batch,'p50_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95)),
                                     'predictions_per_second':batch/float(np.median(times)),
                                     'max_abs_delta':delta,'passed':delta<1e-5})
        rng=np.random.default_rng(731);differences=rng.normal(size=100);indices=rng.integers(0,100,size=(2000,100))
        t=time.perf_counter();cpu_means=differences[indices].mean(axis=1);cpu_seconds=time.perf_counter()-t
        def gpu_statistics():
            dx=torch.tensor(differences,device='cuda',dtype=torch.float64)
            ix=torch.tensor(indices,device='cuda')
            return dx[ix].mean(dim=1).cpu().numpy()
        gpu_means,gpu_seconds=timed(gpu_statistics)
        delta=float(np.max(np.abs(cpu_means-gpu_means)))
        probes['S10']={'cpu_fp64_seconds':cpu_seconds,'cuda_fp64_seconds':gpu_seconds,'max_abs_delta':delta,
                       'same_draw_indices':True,'passed':delta<1e-12,'scope':'fixed independent-index numerical probe; research inference uses calendar blocks'}
        telemetry=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,temperature.gpu,power.draw,memory.used','--format=csv,noheader'],capture_output=True,text=True)
        report={'created_at':now(),'evidence_kind':'systems_benchmark','fixture':'synthetic_development_only','reserved_access':False,
                'device':device,'initial_free_bytes':free,'total_bytes':total,'torch':torch.__version__,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'observable_telemetry':telemetry.stdout.strip() or None,
                'probes':probes,'qualified_research_systems':False,'limitations':['synthetic workload','no process-kill recovery','no thermal time series']}
        atomic_json(repo/'reports/systems_qualification.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
