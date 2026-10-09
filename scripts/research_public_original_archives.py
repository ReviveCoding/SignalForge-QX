"""Preserve exact official development archive bytes, without input promotion."""
import hashlib,json,re
from signalforge.runtime import paths,now,digest,commit_bundle,atomic_json,code_hash,file_hash
from signalforge.public_archive_evidence import URLS,fetch_public,explain_eia_notice

repo,runtime=paths(); records=[]; payloads={}; failures=[]
for name,url in URLS.items():
    try:
        if (repo/'reports/original_archive_evidence_research.json').exists():
            previous=json.loads((repo/'reports/original_archive_evidence_research.json').read_text())
            existing=next((r for r in previous['records'] if r['name']==name and r['url']==url),None)
            if existing:
                from signalforge.runtime import validate_bundle
                directory=runtime/previous['relative_bundle'];validate_bundle(directory)
                payload=(directory/existing['filename']).read_bytes()
                if hashlib.sha256(payload).hexdigest()!=existing['sha256']:raise ValueError('Corrupt immutable archive bytes')
                payloads[existing['filename']]=payload;records.append(existing)
                print(json.dumps({'name':name,'state':'REUSED_CHECKSUM_VERIFIED_ARCHIVE','sha256':existing['sha256']}),flush=True)
                continue
        if name=='bls_cpi' and (repo/'reports/original_archive_evidence_research.json').exists():
            previous=json.loads((repo/'reports/original_archive_evidence_research.json').read_text())
            if any(r['name']==name and 'HTTP 403' in r['reason'] for r in previous['failures']):
                raise PermissionError('Prior HTTP 403 retained; no repeated acquisition attempt')
        payload,content_type=fetch_public(url)
        sha=hashlib.sha256(payload).hexdigest(); filename=name+'-'+sha+'.raw'
        payloads[filename]=payload
        text=payload.decode('utf-8',errors='replace')
        checks={}
        if name=='ishares_2023_distributions':
            if not payload.startswith(b'%PDF'):raise ValueError('Issuer archive is not a PDF')
            checks={'historical_distribution_year':2023,'source_scope':'Issuer tax supplement; evidence only, not original announcement clocks or full action completeness'}
        if name=='sec_uso_split_2020':
            plain=' '.join(re.sub('<[^>]+>',' ',text).split())
            checks={'issuer_split_notice_matches':all(t in plain for t in ['April 22, 2020','April 28, 2020','one-for-eight']),
                    'scope':'Single USO split notice; no other action completeness implied'}
        if name=='eia_notice': checks=explain_eia_notice(text)
        if name=='bls_cpi':
            plain=' '.join(re.sub('<[^>]+>',' ',text).split())
            checks={'original_release_embargo_clock_present':all(t in plain for t in ['8:30 a.m. (ET)','January 12, 2023']),
                    'clock_scope':'2023-01-12 CPI release only; full exact series/version mapping remains required'}
        if name=='fed_indpro':
            plain=' '.join(re.sub('<[^>]+>',' ',text).split())
            checks={'release_date_present':'January 18, 2023' in plain,
                    'clock_scope':'single G17 archived release, not all ALFRED revisions'}
        records.append({'name':name,'url':url,'sha256':sha,'bytes':len(payload),'filename':filename,'content_type':content_type,'checks':checks})
        print(json.dumps({'name':name,'state':'ARCHIVE_BYTES_PRESERVED','sha256':sha,'bytes':len(payload),'checks':checks}),flush=True)
    except (PermissionError,ValueError) as error:
        failures.append({'name':name,'error_type':type(error).__name__,'reason':str(error)})
        print(json.dumps(failures[-1]),flush=True)
    except Exception as error:
        failures.append({'name':name,'error_type':type(error).__name__,'reason':'Bounded network/transport failure; no bypass'})
        print(json.dumps(failures[-1]),flush=True)
summary={'state':'COMPLETED_BOUNDED_ORIGINAL_ARCHIVE_EVIDENCE_RESEARCH','created_at':now(),'records':records,'failures':failures,
         'source_tree_hash':code_hash(repo),'original_source_inputs_changed':False,'tier_a_events_promoted':0,
         'reserved_access':False,'p1_qualified':False,'qualified_for_final':False,
         'remaining':'Full original version-to-feature linkage and release correction histories for all source events; complete actions/ex-pay/split/open publication clocks and same-horizon cash benchmark'}
identity=digest({'records':records,'failures':failures,'script':file_hash(repo/'scripts/research_public_original_archives.py')})
target=runtime/'artifacts/original_archive_evidence'/identity
commit_bundle(target,{**payloads,'summary.json':summary},{'evidence_only':True,'reserved_access':False})
summary['relative_bundle']=str(target.relative_to(runtime));atomic_json(repo/'reports/original_archive_evidence_research.json',summary)
print(json.dumps({'state':summary['state'],'archives':len(records),'blocked':len(failures),'relative_bundle':summary['relative_bundle']},indent=2))
