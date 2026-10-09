"""Bounded official-source documentation fetch, separate from schema certification."""
from signalforge.runtime import paths,atomic_json,now
from signalforge.sources import Acquisition,INDEX

repo,runtime=paths();client=Acquisition(runtime,16*1024**2);records=[]
urls={**INDEX,'fred':'https://fred.stlouisfed.org/docs/api/fred/series_observations.html',
      'prices':'https://www.alphavantage.co/documentation/'}
for source,url in urls.items():
    try:
        receipt=client.cached(url,'html') or client.fetch(url,'html')
        records.append({'source':source,'state':'DOCUMENT_BYTES_VERIFIED','raw':receipt,'source_schema_qualified':False})
    except Exception as e:
        records.append({'source':source,'state':'BLOCKED_AUTH' if isinstance(e,PermissionError) else 'FAILED_SOURCE','reason':str(e)})
atomic_json(repo/'reports/source_document_verification.json',{'created_at':now(),'sources':records})
print('Official documentation verification receipts recorded; source clocks/schema still require qualification.')
