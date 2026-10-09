"""Extract and render historical issuer evidence; never qualify full P1 coverage."""
import json,re,subprocess,hashlib
from signalforge.runtime import paths,validate_bundle,commit_bundle,digest,now,atomic_json,file_hash
repo,runtime=paths();research=json.loads((repo/'reports/original_archive_evidence_research.json').read_text());folder=runtime/research['relative_bundle'];validate_bundle(folder)
record=next(r for r in research['records'] if r['name']=='ishares_2023_distributions');pdf=folder/record['filename']
assert file_hash(pdf)==record['sha256']
code='''import fitz,json,sys
pdf=fitz.open(sys.argv[1]);result=[]
for index in [15,27]:
    page=pdf[index];page.get_pixmap(matrix=fitz.Matrix(.8,.8)).save(sys.argv[2]+str(index)+'.png')
    result.append(dict(page=index+1,text=page.get_text()))
print(json.dumps(result))
'''
images=runtime/'tmp/issuer_evidence_2023_';result=subprocess.run([str(runtime/'tools/pdf_qa_env/bin/python'),'-c',code,str(pdf),str(images)],check=True,capture_output=True,text=True,timeout=30)
pages=json.loads(result.stdout);rows=[]
for page in pages:
    lines=page['text'].splitlines()
    for index,line in enumerate(lines):
        if line.strip() not in {'IEF','TLT'}:continue
        segment=lines[index:index+12];dates=[v for v in segment if re.fullmatch(r'\d\d/\d\d/2023',v.strip())]
        if len(dates)!=3:raise ValueError('Issuer record/ex/pay date ordering not established')
        amount=next((v.strip().replace('$','').replace(',','') for v in segment[segment.index(dates[-1])+1:] if re.fullmatch(r'\d+\.\d+',v.strip())),None)
        if amount is None:raise ValueError('Issuer distribution amount absent')
        rows.append({'asset':line.strip(),'record_date':dates[0],'ex_date':dates[1],'pay_date':dates[2],'distribution_per_share':float(amount),'pdf_page':page['page']})
counts={asset:sum(r['asset']==asset for r in rows) for asset in ['IEF','TLT']}
if counts!={'IEF':12,'TLT':12}:raise ValueError('Expected issuer monthly row coverage not established')
summary={'state':'COMPLETED_PARTIAL_ISSUER_ACTION_EVIDENCE','created_at':now(),'issuer_url':record['url'],'pdf_sha256':record['sha256'],'rows':rows,'counts':counts,'complete_2010_2023_action_coverage':False,'original_announcement_clocks_qualified':False,'p1_qualified':False,'reserved_access':False,'scope':'24 2023 issuer distribution records with actual record/ex/pay dates; tax supplement is retrospective, not contemporaneous announcement evidence; raw open clocks, splits/all assets/all years, cash benchmark and licenses remain independent gates'}
identity=digest(summary);target=runtime/'artifacts/issuer_action_evidence'/identity
files={'summary.json':summary,'extraction.json':pages}
for index in [15,27]:files['page'+str(index+1)+'.png']=(runtime/'tmp'/('issuer_evidence_2023_'+str(index)+'.png')).read_bytes()
commit_bundle(target,files,{'evidence_only':True,'reserved_access':False});summary['relative_bundle']=str(target.relative_to(runtime));atomic_json(repo/'reports/issuer_action_evidence_2023.json',summary)
print(json.dumps({'state':summary['state'],'counts':counts,'pdf_sha256':record['sha256'],'relative_bundle':summary['relative_bundle']},indent=2))
