import json
from signalforge.runtime import paths,atomic_json,now,file_hash,validate_bundle
from signalforge.sources import Acquisition,parse_eia_stocks_pdf
from signalforge.data import auxiliary_rows
repo,runtime=paths();p=repo/'reports/eia_development_acquisition.json';a=json.loads(p.read_text())
validate_bundle(runtime/'artifacts/incidents/eia_2019_duplicate_original_inputs')
page='https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/wpsr_2019_07_03.php'
indices=[i for i,r in enumerate(a['records']) if r['release_page']==page]
assert len(indices)==1,'Exactly one explicit release required'
i=indices[0];r=a['records'][i]
assert r['raw']['metadata']['table']=='4'
assert r['raw']['source_url']=='https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/csv/table4.csv'
assert r['raw']['sha256']=='e8e75163084b1d967d681a80de2f5e5d1b74279ce79ab2dbf6e30e79ab85c51f'
raw=Acquisition(runtime).cached('https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/pdf/table4.pdf','pdf')
ex=json.loads((runtime/'artifacts/source_pdf_extractions'/raw['sha256']/'extraction.json').read_text())
print(ex['text'][:1800])
event=parse_eia_stocks_pdf(ex['layout'],page)
assert event['series']['Commercial (Excluding SPR)']['change']==-1.1
old={**r,'state':'FAILED_SOURCE','reason':'Stale CSV duplicates preceding issue; immutable incident preserved'}
a['records'][i]={**r,'raw':raw,'event':event,'state':'SUCCEEDED','superseded_source_attempt':old,'correction_adr':'research/desktop_study/ADR_008_eia_duplicate_pdf_correction.md','visual_verification':{'verified':True,'image':'reports/source_qa/eia_2019_07_03_table4.png','sha256':file_hash(repo/'reports/source_qa/eia_2019_07_03_table4.png')}}
frame=auxiliary_rows(a['records']);assert not (frame.raw_hash==frame.target_raw_hash).any()
a['source_correction_at']=now();atomic_json(p,a)
incident=json.loads((repo/'reports/source_integrity_incident.json').read_text());incident.update(state='SOURCE_CORRECTED_RECOMPUTATION_PENDING',corrected_raw_sha256=raw['sha256'],source_precision=.1,corrected_rows=len(frame),already_known_target_count_after=0)
atomic_json(repo/'reports/source_integrity_incident.json',incident)
for name in ['auxiliary_development_results.json','auxiliary_development_analysis.json','auxiliary_real_reload_qualification.json']:
    path=repo/'reports'/name
    if path.exists():
        doc=json.loads(path.read_text());doc['state']='INVALID_SOURCE_RECOMPUTATION_REQUIRED';doc['source_integrity_valid']=False;atomic_json(path,doc)
atomic_json(repo/'reports/auxiliary_execution_state.json',{'state':'SOURCE_CORRECTED_RECOMPUTATION_PENDING','ended_at':now(),'final_access':False})
print('Correction complete',event)
