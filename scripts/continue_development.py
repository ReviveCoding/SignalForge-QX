"""Bounded authorized continuation: archive segments, admitted fits, report receipts.

No final access occurs here. The full study remains incomplete when independent
all-track software or source gates remain. No hardware-shutdown auto restart.
"""
import json
import subprocess
import sys
from signalforge.runtime import paths,atomic_json,now
from signalforge.cli import processed_audit
from signalforge.reporting import render,validate_lineage


def main():
    repo,runtime=paths();path=repo/'reports/eia_development_acquisition.json'
    state={'started_at':now(),'state':'RUNNING','maximum_archive_segments':4,'final_access':False,'stages':[]}
    status=repo/'reports/continuation_state.json';atomic_json(status,state)
    for segment in range(4):
        before=json.loads(path.read_text())
        if before['completed']==before['required']:break
        rc=subprocess.run([sys.executable,'scripts/acquire_eia_remaining.py'],timeout=1320).returncode
        after=json.loads(path.read_text())
        state['stages'].append({'stage':'archive','segment':segment,'exit_code':rc,'completed':after['completed'],'required':after['required'],'at':now()})
        atomic_json(status,state);processed_audit(repo);render(repo)
        if rc or after['completed']<=before['completed'] or any(r['state']=='BLOCKED_AUTH' for r in after['records']):break
    acquisition=json.loads(path.read_text())
    if acquisition['completed']==acquisition['required']:
        from signalforge.auxiliary import run
        try:
            rows=run(repo,runtime)
            state['stages'].append({'stage':'auxiliary_development','rows':len(rows),'state':'EXECUTED','at':now()})
        except Exception as exc:
            state['stages'].append({'stage':'auxiliary_development','state':'BLOCKED_OR_FAILED','error_type':type(exc).__name__,'reason':str(exc),'at':now()})
    else:
        state['stages'].append({'stage':'auxiliary_development','state':'BLOCKED_DATA','reason':'Archive incomplete; no shorter favorable panel substituted'})
    state['state']='CHECKPOINT_NOT_FULL_STUDY_COMPLETE';state['ended_at']=now();atomic_json(status,state)
    subprocess.run([sys.executable,'scripts/update_checkpoint.py'],check=True,timeout=60)
    render(repo);validate_lineage(repo)
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
