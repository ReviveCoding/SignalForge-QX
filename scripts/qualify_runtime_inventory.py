import json,subprocess,sys
from signalforge.runtime import paths,atomic_json,now,file_hash
from signalforge.environment import validate_inventory
repo,runtime=paths()
result=subprocess.run([sys.executable,'scripts/audit_runtime.py'],capture_output=True,text=True,check=True,timeout=60)
audit=json.loads(result.stdout);stack=json.loads((repo/'configs/stack_candidate.json').read_text())
qualification=validate_inventory(audit,stack,repo,runtime)
qualification.update(observed_at=now(),stack_spec_sha256=file_hash(repo/'configs/stack_candidate.json'),
    dispatcher_sha256=audit['dispatcher_sha256'],versions=audit['versions'],runtime_filesystem=audit['runtime_filesystem'])
atomic_json(repo/'reports/runtime_inventory_qualification.json',qualification);print(json.dumps(qualification,indent=2))
