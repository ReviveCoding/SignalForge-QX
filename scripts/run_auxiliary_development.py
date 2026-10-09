"""Run all admitted Auxiliary development work and persist truthful evidence."""
import json,subprocess,sys
from signalforge.runtime import paths,atomic_json,now
from signalforge.auxiliary import run
from signalforge.reporting import render,validate_lineage
from signalforge.cli import processed_audit

repo,runtime=paths()
state={'started_at':now(),'state':'RUNNING','final_access':False,'authoritative_full_study':False}
atomic_json(repo/'reports/auxiliary_execution_state.json',state)
try:
    if '--cpu-only' in sys.argv:
        from signalforge.development import run_auxiliary
        rows=run_auxiliary(repo,runtime,scope='cpu_only')
    else:rows=run(repo,runtime)
    state.update(state='DEVELOPMENT_CHECKPOINT',result_rows=len(rows))
except Exception as exc:
    state.update(state='BLOCKED_OR_FAILED',error_type=type(exc).__name__,reason=str(exc))
except KeyboardInterrupt:
    state.update(state='INTERRUPTED',reason='Project run interrupted; durable artifacts and charged budget retained')
    raise
finally:
    state['ended_at']=now();atomic_json(repo/'reports/auxiliary_execution_state.json',state)
    subprocess.run([sys.executable,'scripts/update_checkpoint.py'],check=True,timeout=60)
    processed_audit(repo);render(repo);validate_lineage(repo)
    print(json.dumps(state),flush=True)
