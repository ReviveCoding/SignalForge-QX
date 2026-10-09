import json,sys
from signalforge.runtime import paths,atomic_json,validate_bundle,digest,now
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences,FAMILIES
from signalforge.panels import track_folds
from signalforge.development import trial_grid
from signalforge.input_identity import canonical_data_id
from signalforge.correction_cache import corrected_cache
repo,runtime=paths();a=json.loads((repo/'reports/eia_development_acquisition.json').read_text());frame,x=sequences(auxiliary_rows(a['records']))
study=json.loads((repo/'configs/study.json').read_text());folds,blocked=track_folds(frame,study,'Auxiliary-C');assert not blocked
identity=canonical_data_id(a,frame,x);parents,proof_id=corrected_cache(repo,runtime,folds,x,identity)
required=[digest({'partition':'inner_validation','year':year,'family':family,'seed':seed,'trial':trial_grid(family)[-1]}) for year in [2018,2019] for family in FAMILIES for seed in [11,37,71]]
result={'state':'VERIFIED_CANONICAL_CACHE_UPGRADE','canonical_data_id':identity,'proof_id':proof_id,'exact_input_reusable_recipes':len(parents),'registered_pilots_reusable':sum(k in parents for k in required),'registered_pilots_required':len(required),'at':now(),'reserved_access':False,'new_model_fits':0}
if '--require-complete' in sys.argv:
    document=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
    assert document['state']=='SUCCEEDED_DIAGNOSTIC' and document['protocol']['compute_scope']=='all'
    required+= [digest({'partition':'inner_validation','year':year,'family':family,'seed':11,'trial':trial}) for year in document['protocol']['outer_years'] for family in FAMILIES for trial in trial_grid(family)]
    required+= [digest({'partition':'outer_development','year':row['outer_year'],'family':row['family'],'seed':row['seed'],'trial':row['trial']}) for row in document['results'] if row.get('state')=='SUCCEEDED']
    required=set(required)
    result.update(all_registered_recipes_required=len(required),all_registered_recipes_reusable=sum(k in parents for k in required),missing_recipe_ids=sorted(required-set(parents)))
atomic_json(repo/'reports/canonical_cache_upgrade_qualification.json',result);print(json.dumps(result));assert all(k in parents for k in required)
