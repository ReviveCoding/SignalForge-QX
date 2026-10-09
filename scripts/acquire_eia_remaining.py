"""Two bounded I/O workers, shared 1/sec initiation limit, single manifest writer."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin
from signalforge.sources import Acquisition,INDEX,Links,discover_eia,parse_eia_stocks
from signalforge.runtime import paths,atomic_json,now,file_hash


def main():
    repo,runtime=paths();manifest=repo/'reports/eia_development_acquisition.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else {'records':[]}
    records={r['release_page']:r for r in previous['records']}
    initial_completed=sum(r['state']=='SUCCEEDED' for r in records.values())
    client=Acquisition(runtime,256*1024**2)
    index=client.cached(INDEX['eia'],'html') or client.fetch(INDEX['eia'],'html')
    links=Links();links.feed(Path(index['path']).read_text(errors='replace'))
    pages=sorted({urljoin(INDEX['eia'],h) for h,_ in links.links if re.search(r'/20(?:1\d|2[0-3])/20\d\d_\d\d_\d\d/wpsr_',urljoin(INDEX['eia'],h))})
    pending=[]
    for page in pages:
        if records.get(page,{}).get('state')=='SUCCEEDED':
            r=records[page]
            if file_hash(r['raw']['path'])!=r['raw']['sha256']:raise ValueError('Corrupt committed archive')
        elif records.get(page,{}).get('state') not in {'FAILED_SOURCE','BLOCKED_AUTH'} and records.get(page,{}).get('attempts',0)<2:
            pending.append(page)
    plan={'created_at':now(),'release_pages':len(pages),'new_pages':len(pending),'segment_max_seconds':1200,
          'source_workers':2,'shared_initiation_rate_per_second':1,'max_new_bytes':512*1024**2,
          'previous_segment_seconds_per_new_issue':previous.get('elapsed_seconds',0)/max(1,previous.get('segment_completed',
               previous.get('completed',0)-(previous.get('required',len(pages))-previous.get('plan',{}).get('new_pages',len(pages))))),
          'reserved_access':False,'scope':'EIA table4 2010–2023 official discovered CSV archive paths; no semantic change'}
    atomic_json(repo/'reports/eia_acquisition_plan.json',plan)
    started=time.monotonic()
    # Separate sessions; independent per-worker 256 MiB limits are below the shared 512 MiB ceiling.
    clients=[Acquisition(runtime,256*1024**2) for _ in range(2)]
    def fetch(page,worker):
        c=clients[worker]
        try:
            html_receipt=c.cached(page,'html') or c.fetch(page,'html')
            url=discover_eia(Path(html_receipt['path']).read_text(errors='replace'),page,'4')
            raw=c.cached(url,'csv') or c.fetch(url,'csv',metadata={'release_page':page,'table':'4'})
            return {'release_page':page,'state':'SUCCEEDED','raw':raw,'event':parse_eia_stocks(raw['path'],page)}
        except Exception as e:
            retryable=isinstance(e,RuntimeError) and ('Network failure' in str(e) or re.fullmatch(r'HTTP 5\d\d',str(e)))
            return {'release_page':page,'state':'BLOCKED_AUTH' if isinstance(e,PermissionError) else 'FAILED_RETRYABLE' if retryable else 'FAILED_SOURCE','reason':str(e)}
    iterator=iter(pending);jobs={};stop=False
    with ThreadPoolExecutor(max_workers=2) as pool:
        for worker in range(2):
            page=next(iterator,None)
            if page:jobs[pool.submit(fetch,page,worker)]=worker
        while jobs:
            finished,_=wait(jobs,return_when=FIRST_COMPLETED,timeout=45)
            if not finished:
                print('Acquisition workers active; bounded requests pending',flush=True)
            for future in finished:
                worker=jobs.pop(future);record=future.result()
                record['attempts']=records.get(record['release_page'],{}).get('attempts',0)+1
                records[record['release_page']]=record
                completed=sum(r['state']=='SUCCEEDED' for r in records.values())
                atomic_json(manifest,{'plan':plan,'records':[records[k] for k in sorted(records)],'completed':completed,
                                    'required':len(pages),'state':'SUCCEEDED' if completed==len(pages) else 'PARTIAL',
                                    'elapsed_seconds':time.monotonic()-started,'segment_completed':completed-initial_completed})
                print(json.dumps({'completed':completed,'required':len(pages),'state':record['state'],'page':record['release_page']}),flush=True)
                stop=stop or record['state']=='BLOCKED_AUTH' or time.monotonic()-started>=1200
                page=None if stop else next(iterator,None)
                if page:jobs[pool.submit(fetch,page,worker)]=worker
    print('Bounded archive checkpoint committed',flush=True)


if __name__=='__main__':main()
