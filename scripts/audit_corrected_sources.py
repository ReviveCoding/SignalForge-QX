import json
from signalforge.runtime import paths,atomic_json,now
from signalforge.sources import parse_eia_stocks
from signalforge.data import auxiliary_rows
repo,runtime=paths();a=json.loads((repo/'reports/eia_development_acquisition.json').read_text());failures=[]
for record in a['records']:
    if record['state']!='SUCCEEDED':continue
    if record['event'].get('source_format')=='pdf':continue
    try:parse_eia_stocks(record['raw']['path'],record['release_page'])
    except ValueError as e:failures.append({'release':record['release_page'],'reason':str(e)})
frame=auxiliary_rows(a['records'])
receipt={'at':now(),'state':'PASSED_CORRECTED_DEVELOPMENT_SOURCE' if not failures else 'FAILED_SOURCE','csv_rechecks':sum(r['state']=='SUCCEEDED' and r['event'].get('source_format')!='pdf' for r in a['records']),'failures':failures,'rows':len(frame),'known_raw_targets':0,'reserved_access':False,'whole_archive_complete':a['completed']==a['required']}
atomic_json(repo/'reports/eia_corrected_source_audit.json',receipt);print(json.dumps(receipt))
assert not failures
