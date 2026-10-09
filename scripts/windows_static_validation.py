"""Windows-only AST/JSON validation. Does not import research code or enter WSL."""
import ast
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import sys

def main():
    repo=Path(__file__).resolve().parents[1]
    if sys.platform!='win32':raise RuntimeError('This static receipt is for native Windows only')
    files=sorted({p for root in ['src','scripts','tests'] for p in (repo/root).rglob('*.py')})
    errors=[];hashes={};trees={}
    for path in files:
        relative=path.relative_to(repo).as_posix()
        hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        try:trees[relative]=ast.parse(path.read_text(encoding='utf-8-sig'),filename=relative)
        except (SyntaxError,UnicodeError) as error:errors.append({'file':relative,'error':str(error)})
    # Check local package module names without importing any dependency.
    modules={p.stem for p in (repo/'src/signalforge').glob('*.py')}
    for name,tree in trees.items():
        for node in ast.walk(tree):
            target=None
            if isinstance(node,ast.ImportFrom):
                if name.startswith('src/signalforge/') and node.level==1 and node.module:target=node.module.split('.')[0]
                elif node.level==0 and node.module and node.module.startswith('signalforge.'):target=node.module.split('.')[1]
            if target and target not in modules:errors.append({'file':name,'error':'Missing local import module: '+target})
    json_files=sorted([p for root in ['configs','contracts'] for p in (repo/root).rglob('*.json')])
    for path in json_files:
        try:json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError,UnicodeError) as error:errors.append({'file':str(path.relative_to(repo)),'error':str(error)})
    fred=repo/'.local/fred_request_plan.json'
    fred_sha=hashlib.sha256(fred.read_bytes()).hexdigest()
    if fred_sha!='d144daec920dcec77a9fdf51170d468de92c281c092b83f8d43cebc1c8c0ffbb':
        errors.append({'file':'.local/fred_request_plan.json','error':'Previously registered FRED plan bytes changed'})
    tiingo=repo/'.local/tiingo_price_plan.json'
    plan=json.loads(tiingo.read_text(encoding='utf-8-sig'))
    required={'source':'tiingo','partition':'development','reserved_access':False,'request_selection':'outcome_blind',
        'symbols':['SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'],'startDate':'2010-07-20','endDate':'2023-12-31',
        'resampleFreq':'daily','format':'json','max_bytes':16777216,'max_bytes_per_symbol':2097152,
        'max_pages_per_symbol':1,'max_rows_per_symbol':4000,'calendar':'XNYS','price_mode':'P0',
        'basis':'unadjusted_close_price_return',
        'fields':['date','open','high','low','close','volume','adjOpen','adjHigh','adjLow','adjClose','adjVolume','divCash','splitFactor']}
    if set(plan)!=set(required)|{'registered_at'} or any(type(plan.get(k)) is not type(v) or plan.get(k)!=v for k,v in required.items()):
        errors.append({'file':'.local/tiingo_price_plan.json','error':'Frozen secret-free Tiingo plan contract differs'})
    result={'state':'STATIC_VALID' if not errors else 'STATIC_INVALID','created_at':datetime.now(timezone.utc).isoformat(),
        'interpreter':sys.executable,'platform':sys.platform,'python_ast_files':len(files),'config_contract_json_files':len(json_files),
        'source_file_hashes':hashes,'errors':errors,'unit_tests_executed':False,'wsl_entered':False,
        'claim_boundary':'AST/JSON and local module name checks only; no research imports, unit fixtures, CUDA, acquisition or acceptance execution',
        'fred_plan_sha256':fred_sha,'tiingo_plan_sha256':hashlib.sha256(tiingo.read_bytes()).hexdigest()}
    path=repo/'reports/windows_code_only_static_validation.json'
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_file_hashes'},indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
