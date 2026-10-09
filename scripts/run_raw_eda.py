import json
from signalforge.runtime import paths,atomic_json,now
from signalforge.data import eia_events
from signalforge.eda import event_audit
repo,runtime=paths();manifest=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
result=event_audit(eia_events(manifest['records']));result['created_at']=now()
result['unresolved_sources']=[{'release':r['release_page'],'state':r['state'],'reason':r.get('reason')} for r in manifest['records'] if r['state']!='SUCCEEDED']
atomic_json(repo/'reports/eia_raw_eda.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='series'}))
