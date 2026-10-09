import json
from signalforge.runtime import paths
from signalforge.track_source_integration import build_extension
repo,runtime=paths()
print(json.dumps(build_extension(repo,runtime),indent=2))
