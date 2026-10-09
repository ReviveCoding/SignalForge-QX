from signalforge.runtime import paths
from signalforge.cftc_integration import build_canonical
import json
repo,runtime=paths();result=build_canonical(repo,runtime);print(json.dumps({k:v for k,v in result.items() if k!='assets'},sort_keys=True))
