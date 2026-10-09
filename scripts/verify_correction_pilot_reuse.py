import json
from signalforge.runtime import paths,validate_bundle,digest
from signalforge.auxiliary import FAMILIES
from signalforge.development import trial_grid
repo,runtime=paths();audit=json.loads((repo/'reports/source_correction_cache_audit.json').read_text())
p=runtime/'artifacts/source_correction_cache_proofs'/audit['proof_id'];validate_bundle(p);proof=json.loads((p/'proof.json').read_text())
required=[digest({'partition':'inner_validation','year':year,'family':family,'seed':seed,'trial':trial_grid(family)[-1]}) for year in [2018,2019] for family in FAMILIES for seed in [11,37,71]]
print(json.dumps({'pilot_exact_input_reusable':sum(k in proof['parents'] for k in required),'pilot_required':len(required),'partition_proofs':proof['proofs']}))
assert all(k in proof['parents'] for k in required)
