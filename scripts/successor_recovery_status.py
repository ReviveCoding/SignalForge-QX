"""Read-only successor recovery supervision. Unsafe gates never trigger retries."""
import json
from signalforge.runtime import paths
from signalforge.successor import successor_authorized

repo,runtime=paths()
if not successor_authorized(repo):raise PermissionError('AUTH_GATE: successor marker absent')
progress_path=repo/'reports/successor_gpu_progress.json'
progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
incident_path=repo/'reports/successor_gpu_incident.json'
incident=json.loads(incident_path.read_text()) if incident_path.exists() else None
if incident and incident.get('state')!='RESOLVED_VERIFIED':
    result={'state':'MANUAL_BLOCKED','stage':incident.get('stage','SUCCESSOR_GPU'),
            'reason':incident.get('classification','UNKNOWN_UNSAFE')+': '+incident.get('reason','incident requires verified repair'),
            'repair_attempt':incident.get('repair_attempt',0),'incident_path':str(incident_path)}
    result['classification']=incident.get('classification','UNKNOWN_UNSAFE')
    result['incident_id']=incident.get('incident_id','unknown')
else:
    terminal_path=repo/'reports/successor_continuation_terminal.json'
    terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    superseded_manual_block=bool(terminal and terminal.get('state')=='MANUAL_BLOCKED_ADDITIONAL_RUNTIME_REPAIR_AUTHORIZATION'
                                 and incident and incident.get('state')=='RESOLVED_VERIFIED')
    if terminal and not superseded_manual_block:
        result={'state':'DEVELOPMENT_TERMINAL_RESERVED_SEALED','stage':'RESERVED FINAL SEALED',
                'reason':'Development continuation terminal receipt exists; reserved/final gates remain separate.','repair_attempt':0}
    else:
        result={'state':'WATCHING_GPU_AUTHORIZED','stage':progress.get('stage','SUCCESSOR_GPU_AUTHORIZED_WAITING_GATES'),
                'reason':('Verified manual CUDA repair supersedes the preserved manual-block checkpoint; registered successor gates remain active.'
                          if superseded_manual_block else
                          'Registered successor-only authorization; existing resource/validation/budget gates and STOP/PAUSE controls retained.'),
                'repair_attempt':incident.get('repair_attempt',0) if incident else 0}
print(json.dumps({**result,'max_repair_attempts':2,'reserved_access':False}))
