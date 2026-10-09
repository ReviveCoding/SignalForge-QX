"""Materialize exact-input fitted bundles under stronger keys; forbid every fit."""
import json,sqlite3
from signalforge.runtime import paths,atomic_json,now
from signalforge import development
repo,runtime=paths()
qualification=json.loads((repo/'reports/canonical_cache_upgrade_qualification.json').read_text())
if qualification.get('all_registered_recipes_required')!=qualification.get('all_registered_recipes_reusable') or qualification.get('missing_recipe_ids'):
    raise RuntimeError('BLOCKED_CACHE_PROOF: every registered recipe must have an exact-input parent')
if not qualification.get('all_registered_recipes_required'):raise RuntimeError('Complete cache preflight absent')
connection=sqlite3.connect('file:'+str(runtime/'ledger/development_compute.sqlite')+'?mode=ro',uri=True)
def charges():return connection.execute('SELECT COALESCE(SUM(seconds),0),COUNT(*) FROM compute_charges').fetchone()
before=charges()
def forbid_new_training(*args,**kwargs):raise RuntimeError('BLOCKED_CACHE_PROOF: missing validated parent; new training forbidden in materialization')
forbid_new_training.cache_only_training_forbidden=True
development.fit_artifact=forbid_new_training
rows=development.run_auxiliary(repo,runtime,cache_only=True)
after=charges();connection.close()
if before!=after:raise RuntimeError('Unexpected compute charge in cache-only stage')
if any(r['state']!='SUCCEEDED' for r in rows):raise RuntimeError('Cache materialization incomplete; failed recipes retained')
result={'state':'SUCCEEDED_CACHE_ONLY_MATERIALIZATION','at':now(),'new_model_fits':0,'new_compute_charges':0,
        'outer_bundles':len(rows),'canonical_data_id':qualification['canonical_data_id'],'proof_id':qualification['proof_id'],'reserved_access':False}
atomic_json(repo/'reports/canonical_materialization.json',result);print(json.dumps(result))
