import json

from signalforge.auxiliary import sequences
from signalforge.data import auxiliary_rows
from signalforge.input_identity import snapshot_inputs
from signalforge.runtime import digest, paths
from signalforge.stage_cache import training_dependencies

repo, runtime = paths()
a = json.loads((repo / "reports/eia_development_acquisition.json").read_text())
frame, x = sequences(auxiliary_rows(a["records"]))
identity = snapshot_inputs(runtime, a, frame, x, digest(training_dependencies(repo)))
print("Corrected canonical input snapshot", identity)
