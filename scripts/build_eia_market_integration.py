import json

from signalforge.eia_market_integration import build_canonical
from signalforge.runtime import paths

repo, runtime = paths()
result = build_canonical(repo, runtime)
print(json.dumps(result, sort_keys=True))
