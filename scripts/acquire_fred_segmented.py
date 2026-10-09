import json
from signalforge.runtime import paths
from signalforge.source_requests import keyed_pilot
repo,runtime=paths()
print(json.dumps(keyed_pilot(repo,runtime,'fred'),indent=2))
