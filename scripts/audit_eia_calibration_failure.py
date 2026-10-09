"""Evidence-bearing rejection of an inconsistent same-release PDF substitution."""
import csv,json,re
from pathlib import Path
from signalforge.runtime import paths,atomic_json,commit_bundle,digest,file_hash,now,validate_bundle
from signalforge.sources import Acquisition,parse_eia_stocks,parse_eia_stocks_pdf

repo,runtime=paths();base='https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_28/'
page=base+'wpsr_2023_12_28.php';client=Acquisition(runtime)
csv_raw=client.cached(base+'csv/table4.csv','csv');pdf_raw=client.cached(base+'pdf/table4.pdf','pdf')
if not csv_raw or not pdf_raw:raise RuntimeError('BLOCKED_DATA: actual cached same-release sources required')
for raw in [csv_raw,pdf_raw]:
    if file_hash(raw['path'])!=raw['sha256']:raise ValueError('Source raw changed')
extraction=runtime/'artifacts/source_pdf_extractions'/pdf_raw['sha256'];validate_bundle(extraction)
layout=json.loads((extraction/'extraction.json').read_text())['layout']
fields=['Commercial (Excluding SPR)','Total Motor Gasoline','Distillate Fuel Oil','Cushing','SPR']
csv_rows={row[0]:row for row in csv.reader(Path(csv_raw['path']).read_text().splitlines())}
checks=[]
for field in fields:
    row=csv_rows[field];current,prior,change=[float(v.replace(',','')) for v in row[1:4]]
    checks.append({'format':'csv','field':field,'current':current,'prior':prior,'reported_change':change,
        'level_difference_minus_reported_change':current-prior-change,'tolerance':.0021})
    rows=[line for line in layout.splitlines() if re.match(r'^\s*'+re.escape(field)+r'(?:\s|\d|\.)',line)]
    if len(rows)!=1:raise ValueError('PDF field ambiguity')
    values=re.findall(r'(?<![\d.])-?\d[\d,]*\.\d(?!\d)',rows[0].split(field,1)[1])
    current,prior,change=[float(v.replace(',','')) for v in values[:3]]
    checks.append({'format':'pdf','field':field,'current':current,'prior':prior,'reported_change':change,
        'level_difference_minus_reported_change':current-prior-change,'tolerance':.1501})
failures={}
for name,fn,arg in [('csv',parse_eia_stocks,Path(csv_raw['path'])),('pdf',parse_eia_stocks_pdf,layout)]:
    try:fn(arg,page)
    except ValueError as error:failures[name]=str(error)
if set(failures)!={'csv','pdf'}:raise ValueError('Expected actual registered-contract failures absent')
commercial={c['format']:c for c in checks if c['field']=='Commercial (Excluding SPR)'}
if any(abs(c['level_difference_minus_reported_change'])<=c['tolerance'] for c in commercial.values()):raise ValueError('Actual arithmetic failure absent')
image=repo/'reports/source_qa/eia_2023_12_28_table4.png'
result={'state':'BLOCKED_SOURCE_CONSISTENCY','release_page':page,'CSV':csv_raw,'PDF':pdf_raw,
    'checks':checks,'registered_parser_failures':failures,'substitution_admitted':False,
    'visual_verification':{'image':str(image.relative_to(repo)),'sha256':file_hash(image),'verified':True},
    'ADR':'research/desktop_study/ADR_012_eia_2023_source_consistency.md','core_inputs_changed':False,
    'reserved_access':False,'created_at':now(),
    'claim_boundary':'Both official formats fail current arithmetic contract; adjustment/revision cause is unknown. No tolerance widening, invented source correction or scientific qualification.'}
identity=digest(result);commit_bundle(runtime/'artifacts/eia_calibration_failure_audits'/identity,{'audit.json':result},
    {'source_hashes':[csv_raw['sha256'],pdf_raw['sha256']]})
result['artifact_id']=identity;atomic_json(repo/'reports/eia_2023_calibration_source_audit.json',result)
print(json.dumps({'state':result['state'],'artifact_id':identity,'commercial_checks':commercial},indent=2))
