import json

from signalforge.nport_bulk import acquire_bulk
from signalforge.runtime import paths

repo, runtime = paths()
print(json.dumps(acquire_bulk(repo, runtime), indent=2))
