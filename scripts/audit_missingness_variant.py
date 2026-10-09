"""Inspect actual saved development outputs; never reinterpret inactive decay."""
import ast,json
import numpy as np
from signalforge.runtime import paths,validate_bundle,digest,atomic_json,commit_bundle,now,file_hash
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences

repo,runtime=paths();core=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
if core.get('state')!='SUCCEEDED_DIAGNOSTIC':raise RuntimeError('Completed core required')
left=[r for r in core['results'] if r['family']=='gru'];right=[r for r in core['results'] if r['family']=='gru_d']
if len(left)!=15 or len(right)!=15:raise ValueError('Full five-fold three-seed pair required')
checks=[]
tree=ast.parse((repo/'src/signalforge/auxiliary.py').read_text())
current_function=ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_predict'),include_attributes=False)
for reference in left:
    paired=next(r for r in right if r['outer_year']==reference['outer_year'] and r['seed']==reference['seed'])
    values=[];receipts=[]
    for row in [reference,paired]:
        path=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(path);receipts.append(digest(receipt))
        dependency=runtime/'artifacts/model_dependency_manifests'/(receipt['metadata']['code_hash']+'-v2')
        validate_bundle(dependency);stored=json.loads((dependency/'dependencies.json').read_text())
        if stored['dependencies']['functions']['auxiliary.fit_predict']!=current_function:
            raise ValueError('Historical training adapter differs; inspect its frozen dependency record before claiming inactive decay')
        values.append(json.loads((path/'predictions.json').read_text()))
    a,b=values
    if a['decision_time']!=b['decision_time'] or a['y']!=b['y'] or reference['normalizer_id']!=paired['normalizer_id']:
        raise ValueError('Missingness comparison grid/targets/normalizer mismatch')
    checks.append({'year':reference['outer_year'],'seed':reference['seed'],'same_selected_recipe':reference['trial']==paired['trial'],
        'receipt_ids':receipts,'max_abs_output_delta':max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ['mean','quantiles'])})
acquisition=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
frame,raw=sequences(auxiliary_rows(acquisition['records']))
result={'state':'COMPLETED_INACTIVE_DECAY_AUDIT','checks':checks,'raw_context_missing_values':int(np.isnan(raw).sum()),
    'training_adapter_sha256':file_hash(repo/'src/signalforge/auxiliary.py'),
    'registered_core_adapter':'observed tensors are all true; elapsed tensors are all zero in fit_predict',
    'decay_value_identified':False,'interpretation':'Missingness decay is inactive in this core path. Small actual output differences are retained; they do not identify a missingness-decay effect.',
    'generic_track_correction':'Generic track software now uses actual_context_days_v1 and requires ordered context clocks at inference. No public Main/Nested fits exist; corrected software does not change these historical Auxiliary fits.',
    'new_fits':0,'reserved_access':False,'created_at':now()}
identity=digest(result);commit_bundle(runtime/'artifacts/missingness_variant_audits'/identity,{'audit.json':result},{'core_results_sha256':file_hash(repo/'reports/auxiliary_development_results.json')})
result['artifact_id']=identity;atomic_json(repo/'reports/missingness_variant_audit.json',result);print(json.dumps(result,indent=2))
