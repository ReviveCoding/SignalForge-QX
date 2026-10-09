import json
from signalforge.runtime import paths
from signalforge.macro_integration import build_macro_integration
repo,runtime=paths()
print(json.dumps(build_macro_integration(repo,runtime),indent=2))
