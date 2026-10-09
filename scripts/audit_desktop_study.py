"""Audit actual local desktop contracts and immutable attribution references."""
import json
from signalforge.runtime import paths,atomic_json,file_hash,digest,commit_bundle,now
from signalforge.desktop_study import validate_study

repo,runtime=paths();root=repo/'research/desktop_study'
names=['question_contract.json','literature_matrix.json','data_dictionary.json','source_access_matrix.json','mechanism_cards.json','method_source_decisions.json']
documents=[json.loads((root/n).read_text()) for n in names]
result=validate_study(*documents)
for row in documents[-1]:
    path=(root/row['adr']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing or escaping decision ADR')
files={n:file_hash(root/n) for n in names}
for row in documents[-1]:files[row['adr']]=file_hash(root/row['adr'])
result.update(files=files,created_at=now(),scope='Local definition/completeness and semantic audit; not external fact, access or research-data qualification')
identity=digest(result);commit_bundle(runtime/'artifacts/desktop_study_audits'/identity,{'audit.json':result}, {'desktop_study_id':digest(files)})
result['artifact_id']=identity;atomic_json(repo/'reports/desktop_study_qualification.json',result)
print(json.dumps(result,indent=2))
