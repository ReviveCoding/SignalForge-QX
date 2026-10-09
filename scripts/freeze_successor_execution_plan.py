from signalforge.runtime import paths
from signalforge.successor import freeze_execution_plan
import json
repo,runtime=paths();print(json.dumps(freeze_execution_plan(repo,runtime),indent=2))
