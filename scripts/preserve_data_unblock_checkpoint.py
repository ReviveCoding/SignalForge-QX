"""Archive the read-only starting checkpoint before refreshing mutable reports."""
from signalforge.runtime import paths,commit_bundle,file_hash,digest
repo,runtime=paths()
files=list((repo/'reports').glob('*.json'))+list((repo/'reports').glob('technical_report.*'))
files += [repo/'IMPLEMENTATION_STATUS.json',repo/'EXECUTION_PLAN.md',repo/'.local/fred_request_plan.json']
identity=digest({str(p.relative_to(repo)):file_hash(p) for p in files})
commit_bundle(runtime/'artifacts/data_unblock_prior_checkpoints'/identity,
              {p.name:p.read_bytes() for p in files},{'reserved_access':False,'preserve_prior_reports':True})
print(identity)
