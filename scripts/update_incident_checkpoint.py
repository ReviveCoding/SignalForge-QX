import json
from signalforge.runtime import paths,atomic_json,now
repo,runtime=paths();path=repo/'reports/source_integrity_incident.json';doc=json.loads(path.read_text())
doc['state']='CORRECTED_CPU_COMPLETE_ALL_FAMILY_RECOMPUTING';doc['corrected_input_snapshot_id']='6b600f383f2fe0977c63e9a370105720b5350a0e66e156b571b438e6066ef3a4'
doc['cpu_corrected_outer_bundles']=75;doc['cpu_reload_verified']=75;doc['cpu_outage_diagnostics']='reports/auxiliary_cpu_robustness.json';doc['updated_at']=now()
atomic_json(path,doc)
