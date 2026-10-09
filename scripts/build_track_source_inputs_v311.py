import json
from signalforge.runtime import paths
from signalforge.track_source_integration import build_extension_v311
repo,runtime=paths()
print(json.dumps(build_extension_v311(repo,runtime),indent=2))
