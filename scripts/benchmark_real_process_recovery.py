"""Bounded own-process interruption on one actual fixed MLP workload."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,digest,file_hash,commit_bundle,validate_bundle,now,ensure_gpu_owner


def data(repo,runtime):
    from signalforge.data import auxiliary_rows
    from signalforge.auxiliary import sequences
    from signalforge.panels import track_folds
    from signalforge.development import preprocess
    core=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
    if core['state']!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('BLOCKED_DATA: completed core required')
    row=next(r for r in core['results'] if r['family']=='mlp' and r['outer_year']==2022 and r['seed']==11)
    directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
    acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
    frame,raw=sequences(auxiliary_rows(acquisition['records']))
    folds,blocked=track_folds(frame,json.loads((repo/'configs/study.json').read_text()),'Auxiliary-C')
    if blocked:raise RuntimeError('BLOCKED_DATA: original full fold unavailable')
    fold=folds[2022];arrays=preprocess(raw,fold['train'],fold['test'],pd.Timestamp('2022-01-01T00:00Z')-pd.Timedelta(nanoseconds=1))
    tx,vx,scale,normalizer,transform=arrays
    if scale!=row['scale'] or normalizer!=row['normalizer_id'] or transform.manifest()!=json.loads((directory/'transform.json').read_text()):
        raise ValueError('Actual recovery workload transform/normalizer mismatch')
    return row,receipt,fold,tx,vx,scale,directory


def worker(mode,task):
    import torch
    from signalforge.resources import research_gpu_stage
    from signalforge.models import NeuralCUDA
    import signalforge.checkpoint as checkpoint
    torch.set_num_threads(4)
    repo,runtime=paths();task=Path(task).resolve()
    if not task.is_relative_to(runtime/'artifacts/real_process_recovery'):raise PermissionError('Owned recovery task path required')
    protocol=json.loads((task/'protocol.json').read_text())
    row,receipt,fold,tx,vx,scale,directory=data(repo,runtime)
    if digest(receipt)!=protocol['model_receipt_id']:raise ValueError('Recovery reference changed')
    destination=task/mode
    if (destination/'receipt.json').exists():validate_bundle(destination);return
    model=NeuralCUDA('mlp',width=row['trial']['neural'][0],lr=row['trial']['neural'][1],epochs=100,seed=11)
    checkpoint_path=task/'checkpoint.pt'
    original=checkpoint.save_checkpoint
    if mode=='interrupted':
        def stop_after_boundary(path,state,identity):
            original(path,state,identity)
            if state['step']==50:
                atomic_json(task/'ready.json',{'pid':os.getpid(),'step':50,'sample_cursor':state['sample_cursor'],
                    'identity':identity,'worker_script_sha256':file_hash(Path(__file__)),'ready_at':now()})
                limit=time.monotonic()+30
                while time.monotonic()<limit:time.sleep(.05)
                raise RuntimeError('Own-process interruption window expired; no indefinite wait')
        checkpoint.save_checkpoint=stop_after_boundary
    with research_gpu_stage(repo,runtime,'real_process_recovery_'+mode,60) as admission:
        saved=NeuralCUDA.load(directory/'model.bin');reference=json.loads((directory/'predictions.json').read_text())
        expected=saved.predict(vx)
        delta=max(float(np.max(np.abs(a-np.asarray(b)))) for a,b in zip(expected,[reference['mean'],reference['quantiles']]))
        if delta>2e-6:raise ValueError('Actual model/input recovery reload mismatch')
        del saved
        model.fit(tx,fold['train'].y.to_numpy(),scale,observed=np.ones_like(tx,dtype=bool),
            checkpoint_path=checkpoint_path if mode!='baseline' else None,checkpoint_seconds=0)
        ensure_gpu_owner();mean,q=model.predict(vx)
        model_path=runtime/'tmp'/('recovery-model-'+str(os.getpid())+'.pt');model.save(model_path)
        result={'mode':mode,'actual_CUDA_used':True,'epochs':100,'losses':model.losses,
            'mean':mean.tolist(),'quantiles':q.tolist(),'n_training_dates':int(fold['train'].decision_time.nunique()),
            'n_test_dates':int(fold['test'].decision_time.nunique()),'resource_admission':admission,'CUDA_device':torch.cuda.get_device_name(0),
            'source_model_receipt_id':digest(receipt),'reference_reload_max_abs_delta':delta,'completed_at':now()}
        try:commit_bundle(destination,{'model.bin':model_path.read_bytes(),'result.json':result},{'protocol_id':digest(protocol)})
        finally:model_path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['baseline','interrupted','resumed']);parser.add_argument('--task');args=parser.parse_args()
    if args.worker:return worker(args.worker,args.task)
    repo,runtime=paths();row,receipt,fold,_,_,_,_=data(repo,runtime)
    protocol={'experiment':'E10','representative':{'family':'mlp','outer_year':2022,'seed':11},'epochs':100,
        'interrupted_after_step':50,'optimizer':'same AdamW; constant LR, full deterministic batch',
        'model_receipt_id':digest(receipt),'data_id':row['data_id'],'normalizer_id':row['normalizer_id'],
        'measured_matching_fit_seconds':row['seconds'],'maximum_worker_GPU_reservation_seconds':60,
        'maximum_controller_wall_seconds':225,'maximum_own_pause_seconds':30,'raw_output_tolerance':2e-6,
        'implementation_id':file_hash(Path(__file__)),'HPO_new_trials':0,'pilot_new_fits':0,'promotion_eligible':False,'reserved_access':False}
    task=runtime/'artifacts/real_process_recovery'/digest(protocol);task.mkdir(parents=True,exist_ok=True)
    if (task/'receipt.json').exists():
        validate_bundle(task);result=json.loads((task/'result.json').read_text());atomic_json(repo/'reports/real_process_recovery_qualification.json',result);return
    atomic_json(task/'protocol.json',protocol);atomic_json(repo/'reports/real_process_recovery_plan.json',protocol)
    script=str(Path(__file__).resolve())
    def command(mode):return [sys.executable,script,'--worker',mode,'--task',str(task)]
    def run(mode):
        with (task/(mode+'.log')).open('a') as stream:
            stream.write('\nATTEMPT '+now()+'\n');stream.flush()
            subprocess.run(command(mode),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
    try:
        ensure_gpu_owner();run('baseline')
        interruption=task/'interruption_boundary'
        if not (interruption/'receipt.json').exists():
            # Unadmitted partial attempts are preserved before a new controlled
            # interruption; never silently overwrite an old checkpoint.
            leftovers=[p for p in [task/'checkpoint.pt',task/'checkpoint.receipt.json',task/'ready.json'] if p.exists()]
            if leftovers:
                quarantine=task/'unadmitted_attempts'/digest({'at':now(),'files':{p.name:file_hash(p) for p in leftovers}})
                commit_bundle(quarantine,{p.name:p.read_bytes() for p in leftovers},{'reason':'No committed SIGTERM interruption receipt'})
                for path in leftovers:path.unlink()
            (task/'ready.json').unlink(missing_ok=True)
            with (task/'interrupted.log').open('a') as stream:
                stream.write('\nATTEMPT '+now()+'\n');stream.flush()
                proc=subprocess.Popen(command('interrupted'),cwd=repo,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+60
                    while not (task/'ready.json').exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.1)
                    if not (task/'ready.json').exists():raise RuntimeError('Own worker failed before committed interruption boundary')
                    ready=json.loads((task/'ready.json').read_text())
                    import psutil
                    actual=psutil.Process(proc.pid)
                    if ready['pid']!=proc.pid or actual.cmdline()!=command('interrupted') or Path(actual.cwd()).resolve()!=repo.resolve():
                        raise PermissionError('Exact own recovery worker identity mismatch; no unrelated process targeted')
                    if ready['worker_script_sha256']!=file_hash(Path(__file__)):raise PermissionError('Recovery script changed while running')
                    checkpoint_receipt=json.loads((task/'checkpoint.receipt.json').read_text())
                    if checkpoint_receipt['identity']!=ready['identity'] or checkpoint_receipt['optimizer_step']!=50 or file_hash(task/'checkpoint.pt')!=checkpoint_receipt['sha256']:
                        raise ValueError('Interruption boundary not actually committed')
                    proc.terminate();exit_code=proc.wait(timeout=15)
                    if exit_code!=-15:raise RuntimeError('Own worker was not actually SIGTERM interrupted')
                    commit_bundle(interruption,{'checkpoint.bin':(task/'checkpoint.pt').read_bytes(),
                        'checkpoint_receipt.json':checkpoint_receipt,'ready.json':ready,'signal.json':{'signal':'SIGTERM','exit_code':exit_code,'actual_at':now()}},
                        {'owned_worker_command':command('interrupted'),'owned_worker_pid':proc.pid})
                finally:
                    if proc.poll() is None:proc.kill();proc.wait(timeout=15)
            # Only own worker/context was terminated. Wait boundedly for its
            # inventory disappearance; unknown/foreign owners still block.
            for attempt in range(20):
                try:ensure_gpu_owner();break
                except RuntimeError:
                    if attempt==19:raise
                    time.sleep(.1)
        validate_bundle(interruption)
        run('resumed')
        baseline=json.loads((task/'baseline/result.json').read_text());resumed=json.loads((task/'resumed/result.json').read_text())
        delta=max(float(np.max(np.abs(np.asarray(baseline[k])-np.asarray(resumed[k])))) for k in ['mean','quantiles'])
        if baseline['losses']!=resumed['losses'] or delta>2e-6:raise ValueError('Interrupted actual workload differs from uninterrupted reference')
        import torch
        # CPU inspection compares stored parameter tensors, never runs a neural
        # model on CPU. Both training/prediction workers used actual CUDA.
        a=torch.load(task/'baseline/model.bin',map_location='cpu',weights_only=False)['model']
        b=torch.load(task/'resumed/model.bin',map_location='cpu',weights_only=False)['model']
        parameter_delta=max(float(torch.max(torch.abs(a[k]-b[k]))) for k in a)
        if parameter_delta>2e-6:raise ValueError('Actual recovery parameter mismatch')
        ready=json.loads((interruption/'ready.json').read_text())
        result={'state':'PASSED_ACTUAL_DEVELOPMENT_OWN_PROCESS_RECOVERY','protocol':protocol,
            'actual_own_process_SIGTERM':True,'interrupted_step':ready['step'],'sample_cursor':ready['sample_cursor'],
            'final_steps':100,'n_distinct_training_dates':baseline['n_training_dates'],'n_distinct_test_dates':baseline['n_test_dates'],
            'max_prediction_abs_delta':delta,'max_parameter_abs_delta':parameter_delta,'loss_traces_exactly_equal':True,
            'actual_CUDA_used':True,'qualified_research_systems':False,'reserved_access':False,'created_at':now(),
            'limitations':['one fixed actual MLP workload','no OS shutdown or driver recovery','replicas are not scientific candidates']}
        commit_bundle(task,{'result.json':result,'frozen_protocol.json':protocol,'interruption.json':ready},
            {'baseline_receipt_id':digest(validate_bundle(task/'baseline')),'resumed_receipt_id':digest(validate_bundle(task/'resumed')),
             'interruption_receipt_id':digest(validate_bundle(interruption))})
    except Exception as error:
        result={'state':'BLOCKED_OR_FAILED_REAL_PROCESS_RECOVERY','error_type':type(error).__name__,'reason':str(error),
            'protocol':protocol,'owned_partial_artifacts':str(task),'reserved_access':False,'created_at':now()}
        atomic_json(task/'failure.json',result)
        commit_bundle(task/'failed_attempts'/digest(result),{'failure.json':result},{'protocol_id':digest(protocol)})
    atomic_json(repo/'reports/real_process_recovery_qualification.json',result);print(json.dumps(result,indent=2))
    if result['state']!='PASSED_ACTUAL_DEVELOPMENT_OWN_PROCESS_RECOVERY':raise SystemExit(2)


if __name__=='__main__':main()
