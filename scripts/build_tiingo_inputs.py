"""Build real receipt-bound development P0 cards, never fixture/public substitutions."""
import json
from signalforge.runtime import paths,atomic_json,now
from signalforge.tiingo import build_inputs
repo,runtime=paths()
try:result=build_inputs(repo,runtime)
except (ValueError,RuntimeError,PermissionError,FileNotFoundError,KeyError,TypeError) as error:
    result={'state':'BLOCKED_DATA','reason':str(error),'reserved_access':False,
        'qualified_for_final':False,'economic_qualified':False,'created_at':now()}
    atomic_json(repo/'reports/tiingo_input_build.json',result)
print(json.dumps(result,indent=2))
