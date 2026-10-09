"""Resume verified public development archives, with measured bounded acquisition plan."""
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from signalforge.sources import Acquisition,INDEX,Links,discover_eia,parse_eia_stocks
from signalforge.runtime import paths,atomic_json,now,file_hash

repo,runtime=paths()
client=Acquisition(runtime,512*1024**2)
index=client.cached(INDEX['eia'],'html') or client.fetch(INDEX['eia'],'html')
links=Links();links.feed(Path(index['path']).read_text(errors='replace'))
pages=sorted({urljoin(INDEX['eia'],h) for h,label in links.links if re.search(r'/20(?:1\d|2[0-3])/20\d\d_\d\d_\d\d/wpsr_',urljoin(INDEX['eia'],h))})
manifest=repo/'reports/eia_development_acquisition.json'
previous=json.loads(manifest.read_text()) if manifest.exists() else {'records':[]}
done={r['release_page']:r for r in previous['records'] if r['state']=='SUCCEEDED'}
records=dict(done)
plan={'created_at':now(),'release_pages':len(pages),'max_new_bytes':512*1024**2,
      'estimated_request_seconds_from_pilot':10,'estimated_remaining_seconds':20*(len(pages)-len(done)),
      'segment_max_seconds':1200,'source_workers':1,'reserved_access':False,
      'source_scope':'EIA stocks table4 2010-2023; exact official links only',
      'data_use':'Auxiliary-C development; no market economics or all-information claims'}
atomic_json(repo/'reports/eia_acquisition_plan.json',plan)
started=time.monotonic()
for page in pages:
    if page in done:
        if file_hash(done[page]['raw']['path'])!=done[page]['raw']['sha256']:
            raise ValueError('Committed EIA raw artifact corrupt')
        continue
    if time.monotonic()-started>plan['segment_max_seconds']:
        break
    try:
        html_receipt=client.cached(page,'html') or client.fetch(page,'html')
        html=Path(html_receipt['path']).read_text(errors='replace')
        url=discover_eia(html,page,'4')
        raw=client.cached(url,'csv') or client.fetch(url,'csv',metadata={'release_page':page,'table':'4'})
        event=parse_eia_stocks(raw['path'],page)
        records[page]={'release_page':page,'state':'SUCCEEDED','raw':raw,'event':event}
    except Exception as e:
        records[page]={'release_page':page,'state':'BLOCKED_AUTH' if isinstance(e,PermissionError) else 'FAILED_SOURCE','reason':str(e)}
        if isinstance(e,PermissionError):
            break
    completed=sum(r['state']=='SUCCEEDED' for r in records.values())
    atomic_json(manifest,{'plan':plan,'records':list(records.values()),'completed':completed,
                         'required':len(pages),'state':'SUCCEEDED' if completed==len(pages) else 'PARTIAL',
                         'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'completed':completed,'required':len(pages),'page':page,'state':records[page]['state']}),flush=True)
print('Committed bounded acquisition checkpoint',flush=True)
