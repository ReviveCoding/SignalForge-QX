"""Durable successor progress, incidents and independent resource ledgers."""
import json
import subprocess
import traceback
import re
from pathlib import Path
from .runtime import atomic_json,now,code_hash,file_hash,digest,commit_bundle
from .successor import ledger_snapshot,classify_failure


def gpu_context():
    result={}
    for key,query in [('device','--query-gpu=name,utilization.gpu,memory.used,memory.total'),
                      ('compute_apps','--query-compute-apps=pid')]:
        p=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        result[key]={'exit_code':p.returncode,'value':p.stdout.strip()}
    return result


def progress(repo,runtime,stage,**fields):
    ledger_name='successor_gpu_full_compute.sqlite' if stage in {'FULL SUCCESSOR GPU STUDY','SUCCESSOR GPU FULL','TRACK CALIBRATION'} else 'successor_gpu_pilot_compute.sqlite'
    bill=ledger_snapshot(Path(runtime)/'ledger'/ledger_name,21600 if 'full' in ledger_name else 3600)
    recovery_path=Path(repo)/'reports/AUTO_RECOVERY_STATE.json'
    recovery=json.loads(recovery_path.read_text(encoding='utf-8-sig')) if recovery_path.exists() else {}
    doc={'observed_at':now(),'stage':stage,**fields,'ledger':{k:v for k,v in bill.items() if k!='charges'},
         'remaining_gpu_seconds':max(0,(21600 if 'full' in ledger_name else 3600)-bill['charged_seconds']),
         'gpu_context':gpu_context(),'reserved_access':False,'qualified_for_final':False,'repair_attempt':recovery.get('repair_attempt',0)}
    atomic_json(Path(repo)/'reports/successor_gpu_progress.json',doc);return doc


def preserve_incident(repo,runtime,stage,error,**fields):
    repo=Path(repo);runtime=Path(runtime)
    classification=classify_failure(type(error).__name__+': '+str(error))
    incident_id=digest({'stage':stage,'classification':classification,'error_type':type(error).__name__,
                        'track':fields.get('track'),'family':fields.get('family'),
                        'reason_pattern':re.sub(r'\d+(?:\.\d+)?','#',str(error))})
    doc={'state':'OPEN','created_at':now(),'stage':stage,'classification':classification,
         'reason':str(error),'traceback':traceback.format_exc(),'source_tree_hash':code_hash(repo),
         'protocol_sha256':file_hash(repo/'configs/completion_extension_v31.json'),
         'gpu_context':gpu_context(),'repair_attempt':0,'max_repair_attempts':2,
         'ledgers':{name:ledger_snapshot(runtime/'ledger'/name) for name in
                    ['development_compute.sqlite','successor_gpu_pilot_compute.sqlite','successor_gpu_full_compute.sqlite']},
         'reserved_access':False,**fields}
    identity=digest(doc);directory=runtime/'artifacts/successor_gpu_incidents'/identity
    commit_bundle(directory,{'incident.json':doc},{'incident_id':identity,'reserved_access':False})
    atomic_json(repo/'reports/successor_gpu_incident.json',{**doc,'incident_id':incident_id,'snapshot_id':identity,'relative_bundle':str(directory.relative_to(runtime))})
    return doc
