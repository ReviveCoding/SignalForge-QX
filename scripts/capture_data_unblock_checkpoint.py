"""Reconcile the bounded-data continuation without resetting research history."""
import json,sqlite3
from signalforge.runtime import paths,atomic_json,commit_bundle,digest,file_hash,code_hash,now
from signalforge.track_planning import track_cuda_pilot_boundary
from signalforge.source_requests import validate_fred_plan
from signalforge.tiingo import validate_plan
from signalforge.cli import reconcile

repo,runtime=paths()
def read(name):return json.loads((repo/name).read_text(encoding='utf-8-sig'))
status=read('IMPLEMENTATION_STATUS.json')
fred=repo/'.local/fred_request_plan.json'
expected_fred='d144daec920dcec77a9fdf51170d468de92c281c092b83f8d43cebc1c8c0ffbb'
if file_hash(fred)!=expected_fred:raise ValueError('Registered FRED request plan changed; reconcile before continuation')
validate_fred_plan(read('.local/fred_request_plan.json'))
tiingo_id=validate_plan(read('.local/tiingo_price_plan.json'))
with sqlite3.connect((runtime/'ledger/development_compute.sqlite').as_uri()+'?mode=ro',uri=True) as db:
    ceiling=db.execute('SELECT ceiling FROM budget_settings WHERE id=1').fetchone()[0]
    charges=db.execute('SELECT id,seconds,state FROM compute_charges ORDER BY id').fetchall()
    charged=db.execute('SELECT COALESCE(SUM(seconds),0) FROM compute_charges').fetchone()[0]
if ceiling!=43200:raise ValueError('Development budget ceiling changed')
prior=status.get('development_budget_checkpoint',{})
if charged<prior.get('charged_seconds',0) or len(charges)<prior.get('charge_entries',0):raise ValueError('Historical compute charges lost')
budget={'ceiling_seconds':ceiling,'charged_seconds':charged,'remaining_seconds':ceiling-charged,
    'charge_entries':len(charges),'states':{s:sum(row[2]==s for row in charges) for s in sorted({row[2] for row in charges})},
    'charges_digest':digest(charges),'budget_ceiling_changed':False,'observed_at':now(),
    'ledger_scope':'read-only persistent cumulative development ledger; failures and reservations retained'}
raw=reconcile(repo,runtime)
if raw['invalid'] or not raw['final_sealed']:raise ValueError('Raw reconciliation or final seal differs')
pilot=track_cuda_pilot_boundary(repo)
sources=read('reports/keyed_source_access.json')
tracks=read('reports/track_input_qualification.json')
validation=read('reports/test_execution.json')
tests=sum(r['counts']['tests'] for r in validation['results'])
current=all(r['exit_code']==0 and r['source_tree_hash']==code_hash(repo) for r in validation['results'])
record={'state':'BOUNDED_SOURCE_IMPLEMENTATION_RESEARCH_BLOCKED','created_at':now(),'source_tree_hash':code_hash(repo),
    'fred_plan_sha256':file_hash(fred),'tiingo_plan_id':tiingo_id,'credential_presence':sources['credential_presence'],
    'sources':sources['sources'],'tracks':tracks['tracks'],'cuda_pilot_boundary':pilot,'development_budget':budget,
    'raw_reconciliation':raw,'real_new_rows_acquired':0 if not any(s['state'].startswith('SUCCEEDED') for s in sources['sources']) else None,
    'current_full_validation_passed':current,'validation_tests':tests,'auxiliary_successful_outer_fits':status['auxiliary_successful_outer_fits'],
    'qualified_for_final':False,'economic_qualified':False,'reserved_access':False,
    'claim_boundary':'Local fixture correctness and development source plumbing; no Tier A or full-study completion'}
identity=digest(record)
commit_bundle(runtime/'artifacts/data_unblock_checkpoints'/identity,{'checkpoint.json':record},{'reserved_access':False})
record['artifact_id']=identity
atomic_json(repo/'reports/data_unblock_checkpoint.json',record)
status.update(updated_at=now(),development_budget_checkpoint=budget,
    data_unblock_checkpoint={'path':'reports/data_unblock_checkpoint.json','artifact_id':identity,'state':record['state']},
    cuda_track_pilot_boundary=pilot,credential_presence=sources['credential_presence'],
    tiingo_development={'path':'reports/tiingo_input_build.json','state':read('reports/tiingo_input_build.json')['state'],'plan_id':tiingo_id},
    active_validation={'receipt':'reports/test_execution.json','source_hash':code_hash(repo),'state':'SUCCEEDED' if current else 'STALE_OR_FAILED','tests':tests},
    tests_executed=tests,current_tree_full_validation_passed=current,
    resume_command=".\\scripts\\Invoke-SignalForge.ps1 -Action python -Rest @('scripts/acquire_keyed_sources.py')",
    remaining_implementation=[
        'FRED/Tiingo/SEC process credentials absent where reported BLOCKED_AUTH; actual bounded rows and hash-bound P0 manifests required',
        'I1-I4 canonical macro/CFTC/EIA/NPORT source, clock, coverage and mapping integration remains data-blocked',
        'Main/Nested CUDA actual-workload admission blocked separately by exhausted cumulative 120-fit pilot ceiling; no automatic increase',
        'Original-publication/vintage Tier A evidence and complete calibration/source selection/freeze receipts absent',
        'P1 raw-open/action completeness/pay-date/clock audit absent; E09 economic qualification blocked',
        'V3-T11 unseen-entity transport BLOCKED_DATA; V3-T24 optional vectorized-seed RNG DEFERRED_METHOD',
        'Actual operational freeze and future observations required for prospective operation; final remains sealed'])
atomic_json(repo/'IMPLEMENTATION_STATUS.json',status)
print(json.dumps({k:record[k] for k in ['state','artifact_id','current_full_validation_passed','validation_tests','real_new_rows_acquired','credential_presence','development_budget']},indent=2))
