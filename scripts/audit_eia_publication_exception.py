"""Supplemental publication audit; never silently changes the registered corpus."""
import json,re
from pathlib import Path
from signalforge.runtime import paths,atomic_json,commit_bundle,digest,now
from signalforge.sources import Acquisition,discover_eia

repo,runtime=paths();client=Acquisition(runtime,4*1024**2)
main='https://www.eia.gov/petroleum/supply/weekly/archive/2022/2022_06_29/wpsr_2022_06_29.php'
supplement='https://www.eia.gov/petroleum/supply/weekly/archive/2022/2022_06_17_data/wpsr_2022_06_17_data.php'
receipts=[]
for url in [main,supplement]:
    receipt=client.cached(url,'html') or client.fetch(url,'html')
    receipts.append(receipt)
plain=re.sub('<[^>]+>',' ',Path(receipts[0]['path']).read_text())
assert 'published on June 29 includes the data for the week ending June 17' in plain
assert 'would have been published June 23' in plain and 'week ending June 24' in plain
url=discover_eia(Path(receipts[1]['path']).read_text(),supplement,'4')
raw=client.cached(url,'csv') or client.fetch(url,'csv',metadata={'release_page':supplement,'table':'4','special_publication_date':'2022-06-29'})
result={'state':'VERIFIED_OFFICIAL_PUBLICATION_EXCEPTION','created_at':now(),'notice_source':receipts[0],
    'supplemental_page':receipts[1],'supplemental_table4_raw':raw,'first_publication_date':'2022-06-29',
    'reference_weeks':['2022-06-17','2022-06-24'],'publication_precision':'official date; intraday time unverified',
    'backdating_to_2022_06_23_prohibited':True,'registered_main_issue_clock_changed':False,
    'registered_target_changed':False,'supplemental_raw_in_registered_model_inputs':False,
    'archive_count_scope':'646 discovered dated main-issue pages; supplemental reference-week tables are a separate source-accounting scope',
    'qualification':'Reconstructed Tier B; original archived vintage still unverified','reserved_access':False}
identity=digest(result);commit_bundle(runtime/'artifacts/source_publication_exceptions'/identity,{'audit.json':result},{'source':'EIA','reserved_access':False})
result['artifact_id']=identity;atomic_json(repo/'reports/eia_publication_exceptions.json',result)
print(json.dumps(result,indent=2))
