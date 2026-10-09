"""Acquire only the frozen segmented FRED development plan with process-only credentials."""
import json,time
from signalforge.runtime import paths,atomic_json,now
from signalforge.source_requests import keyed_pilot,fred_plan_path,validate_fred_plan
from signalforge.runtime import file_hash

repo,runtime=paths()
path=fred_plan_path(repo)
if not path.exists():
    raise RuntimeError('BLOCKED_REQUEST_PLAN: segmented FRED plan absent')
plan=json.loads(path.read_text(encoding='utf-8-sig'))
plan_id=validate_fred_plan(plan,require_json_safe_segments=True)
before=file_hash(path)

result=None
attempts=[]
for attempt in range(1,4):
    try:
        result=keyed_pilot(repo,runtime,'fred')
        attempts.append({'attempt':attempt,'state':'SUCCEEDED'})
        break
    except RuntimeError as error:
        message=str(error)
        transient=message.startswith('Network failure (') or message.startswith('Network response failure')
        attempts.append({'attempt':attempt,'state':'TRANSIENT_NETWORK_FAILURE' if transient else 'FAILED_NONTRANSIENT',
                         'error_type':type(error).__name__})
        if not transient or attempt==3:
            raise
        # Completed pages are immutable and keyed by credential-free request identity,
        # so the next bounded attempt reuses them and fetches only missing pages.
        time.sleep(15*attempt)
if result is None:
    raise RuntimeError('FRED acquisition ended without a terminal result')
if file_hash(path)!=before:
    raise ValueError('FRED request plan changed during acquisition')
receipt={
    'state':result['state'],
    'source':'fred',
    'plan_id':result.get('plan_id',plan_id),
    'artifact_id':result.get('artifact_id'),
    'observation_versions':result.get('observation_versions'),
    'segmented_plan_sha256':before,
    'acquisition_attempts':attempts,
    'reserved_access':False,
    'created_at':now(),
}
atomic_json(repo/'reports/fred_only_resume.json',receipt)
print(json.dumps(receipt,indent=2))
