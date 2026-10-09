"""Bounded GET-only official historical evidence collection; no model or current data."""
import json,time,urllib.request,urllib.error,urllib.robotparser,hashlib
from pathlib import Path
from urllib.parse import urlsplit
from signalforge.runtime import now,digest,file_hash,commit_bundle,validate_bundle
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering');Q=R/'data_recovery_v36';O=R/'reports/data_recovery_v36';UA='SignalForge-QX-public-documentary/3.6'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def append_progress(stage,**kw):
    v={'observed_at':now(),'stage':stage,'GPU_training':0,'real_forecasts':0,**kw}
    with (O/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v,sort_keys=True)+'\n');f.flush()
    print(json.dumps(v),flush=True)
def roster():
    plan=load(Q/'intent_register_v1.json');p=Q/'official_roster_v1.json'
    if p.exists():return load(p)
    rows=[]
    def add(family,url,purpose,scope='PRE_2024_HISTORICAL_DOCUMENT'):rows.append({'family':family,'url':url,'purpose':purpose,'scope':scope})
    for d in ['2010_07_21','2010_07_28','2023_08_16','2023_08_23','2023_12_20','2023_12_28']:
        stem='https://www.eia.gov/petroleum/supply/weekly/archive/'+d[:4]+'/'+d+'/'
        for kind in ['csv','pdf']:add('EIA',stem+kind+'/table4.'+kind,'Exact stocks Table4 paired formats; original/revision and unit audit')
        add('EIA',stem+'wpsr_'+d+('.html' if d.startswith('2010') else '.php'),'Dated release context and correction notices')
    add('EIA','https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_28/pdf/appendix_e.pdf','Correction scope: propane versus actual amended Table4')
    for name in ['cpi_01122023.htm','cpi_12122023.htm','empsit_01062023.htm','empsit_12082023.htm']:add('BLS','https://www.bls.gov/news.release/archives/'+name,'Original archived CPI/employment release and embargo/correction semantics')
    for d in ['20230118','20231215']:add('FED_G17','https://www.federalreserve.gov/releases/g17/'+d+'/default.htm','Historical INDPRO release, units/revision semantics')
    for family in ['fut_fin_txt_','fut_disagg_txt_']:add('CFTC','https://www.cftc.gov/files/dea/history/'+family+'2023.zip','2023-only annual positioning archive and update ambiguity')
    for typ in ['secured/sofr','unsecured/effr']:add('CASH_NYFED','https://markets.newyorkfed.org/api/rates/'+typ+'/search.json?startDate=2023-01-01&endDate=2023-12-31','Server-bounded 2023 cash rate records; not same-horizon cash execution evidence')
    add('CASH_TREASURY','https://home.treasury.gov/resource-center-data-chart-center/interest-rates/pages/xml?data=daily_treasury_bill_rates&field_tdr_date_value=2023','Official server-bounded 2023 bill quotations and basis semantics')
    for suffix,purpose in [('2023-ishares-distribution-summary-stamped.pdf','Reconfirm exact 2023 IEF/TLT ex/pay/amount tuples'),('ishares-silver-tax-data-12-31-23-stamped.xls','Issuer SLV 2023 silver-sale expense data, not dividend/no-action proof'),('ishares-silver-trust-broker-stamped.pdf','Issuer explicitly 2023 income/expense statement'),('2023-ishares-us-government-source-income-information-stamped.pdf','IEF/TLT 2023 annual income tax categories, not as-of cash rates')]:add('ISSUER','https://www.ishares.com/us/literature/tax-information/'+suffix,purpose)
    value={'registered_at':now(),'parent_plan_id':plan['plan_id'],'rows':rows,'unique_candidates':len(rows),'candidate_selection':'documented historical source templates and official 2023 tax-kit links; no score selection','not_selected':'current product pages, QII with 2024 pay columns, SEC previously403 URL, login or unknown current archives'};value['roster_id']=digest(value)
    with p.open('x') as f:json.dump(value,f,indent=2)
    return value
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None

def collect():
    plan=load(Q/'intent_register_v1.json');reg=roster();opener=urllib.request.build_opener(NoRedirect());attempts=[];ledger=[];robots={};blocked=set();last=0.;total=0
    path=O/'attempted_public_sources.json'
    if path.exists():
        v=load(path)
        for r in v['records']:
            if r.get('relative_bundle'):validate_bundle(T/r['relative_bundle'])
        append_progress('COLLECTION_REUSE_COMPLETE',requests=v['requests'],documents=v['payload_documents']);return v
    def get(url,kind):
        nonlocal last,total
        if any(a['url']==url for a in attempts):raise ValueError('One GET attempt per URL')
        if len(attempts)>=plan['max_http_requests']:raise PermissionError('Registered request cap')
        remaining=plan['max_total_bytes']-total
        if remaining<=0:raise PermissionError('Registered aggregate byte cap')
        delay=plan['min_request_interval_seconds']-(time.monotonic()-last)
        if delay>0:time.sleep(delay)
        last=time.monotonic();a={'url':url,'kind':kind,'started_at':now(),'started_monotonic':last,'attempt':1};attempts.append(a)
        try:
            with opener.open(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=plan['timeout_seconds']) as response:
                payload=response.read(min(plan['max_response_bytes'],remaining)+1);total+=len(payload)
                if len(payload)>plan['max_response_bytes'] or total>plan['max_total_bytes']:raise ValueError('Response/aggregate byte limit exceeded')
                a.update(status=response.status,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),retrieved_at=now(),headers={k:response.headers.get(k) for k in ['Content-Type','ETag','Last-Modified']})
            folder=T/'artifacts/data_recovery_v36/public'/digest(a);commit_bundle(folder,{'payload.bin':payload,'retrieval.json':a},{'evidence_only':True,'reserved_access':False});a['relative_bundle']=str(folder.relative_to(T));return payload,a
        except Exception as error:
            a.update(status=getattr(error,'code',None),state='HTTP_FAILURE' if isinstance(error,urllib.error.HTTPError) else 'PUBLIC_FETCH_FAILED',error_type=type(error).__name__,no_retry=True)
            if a['status'] in [401,403,429]:blocked.add(urlsplit(url).hostname)
            return None,a
    cache=load(R/'reports/qualification_v35/verified_publisher_documents.json')['records'];cache={x['url']:x for x in cache if x.get('relative_bundle')}
    for i,r in enumerate(reg['rows']):
        row={**r,'candidate_number':i+1,'first_public_qualified':False};url=r['url'];host=urlsplit(url).hostname
        if url in cache:
            prior=cache[url];validate_bundle(T/prior['relative_bundle']);folder=T/prior['relative_bundle'];payload=(folder/prior['filename']).read_bytes();assert hashlib.sha256(payload).hexdigest()==prior['sha256'];row.update(state='CACHE_REUSED_VERIFIED_CONTENT',sha256=prior['sha256'],bytes=len(payload),payload_path=str(folder/prior['filename']),cached_receipt_hash=file_hash(folder/'receipt.json'),retrieved_at=prior['retrieved_at']);ledger.append(row);continue
        if host in blocked:row['state']='BLOCKED_PUBLISHER_AUTH_OR_RATE';ledger.append(row);continue
        if host not in robots:
            roboturl='https://'+host+'/robots.txt';body,a=get(roboturl,'robots')
            if body is None and a['status'] not in [404,410]:robots[host]=None
            else:
                parser=urllib.robotparser.RobotFileParser();parser.parse((body or b'').decode('utf-8',errors='replace').splitlines());robots[host]=parser
        if host in blocked or robots[host] is None:row['state']='BLOCKED_ROBOTS_UNAVAILABLE';ledger.append(row);continue
        if not robots[host].can_fetch(UA,url):row['state']='BLOCKED_ROBOTS_DISALLOW';ledger.append(row);continue
        body,a=get(url,'historical_document');row.update({k:v for k,v in a.items() if k not in ['kind','started_monotonic']});row['state']='VERIFIED_CONTENT_ONLY' if body is not None else a['state']
        if body is not None:row['payload_path']=str(T/a['relative_bundle']/'payload.bin')
        ledger.append(row);append_progress('PUBLIC_DOCUMENT_RESULT',candidate=i+1,planned=len(reg['rows']),requests=len(attempts),state=row['state'],family=r['family'])
    out={'created_at':now(),'plan_id':plan['plan_id'],'roster_id':reg['roster_id'],'candidate_urls':len(reg['rows']),'requests':len(attempts),'response_bytes':total,'max_requests':60,'minimum_request_interval_seconds':1.05,'attempts':attempts,'records':ledger,'payload_documents':sum(bool(x.get('payload_path')) for x in ledger),'new_payload_documents':sum(x['state']=='VERIFIED_CONTENT_ONLY' for x in ledger),'first_public_promotions':0,'publisher_stops':sorted(blocked),'no_private_or_403_retries':True}
    with path.open('x') as f:json.dump(out,f,indent=2)
    append_progress('COLLECTION_TERMINAL',requests=len(attempts),payloads=out['payload_documents'],bytes=total);return out
if __name__=='__main__':collect()
