import json
from signalforge.runtime import paths
from signalforge.nport_bulk import acquire_bulk
repo,runtime=paths()
print(json.dumps(acquire_bulk(repo,runtime),indent=2))
