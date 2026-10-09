"""Training-stage cache identity and provenance-checked legacy bundle adoption."""
import ast,json
from pathlib import Path
from .runtime import digest,file_hash,code_hash,commit_bundle,validate_bundle


def training_dependencies(repo):
    repo=Path(repo)
    names=['data','features','models','neural','residual','ssl','pit','metrics','sources']
    env=json.loads((repo/'reports/environment_audit.json').read_text())
    payload={'schema':'training_dependencies_v2',
             'files':{n:file_hash(repo/'src/signalforge'/(n+'.py')) for n in names},
             'functions':{},'study_splits_hpo':{k:json.loads((repo/'configs/study.json').read_text())[k] for k in ['splits','hpo']},
             'environment_id':digest({k:env[k] for k in ['resolved_packages','python','dispatcher_sha256']})}
    selected={'auxiliary':['sequences','fit_predict'],'development':['preprocess','trial_grid','fit_artifact']}
    trees={m:ast.parse((repo/'src/signalforge'/(m+'.py')).read_text()) for m in selected}
    for module,functions in selected.items():
        for name in functions:
            node=next(a for a in trees[module].body if isinstance(a,ast.FunctionDef) and a.name==name)
            payload['functions'][module+'.'+name]=ast.dump(node,include_attributes=False)
    function=next(a for a in trees['development'].body if isinstance(a,ast.FunctionDef) and a.name=='run_auxiliary')
    payload['split_orchestration']=[ast.dump(a,include_attributes=False) for a in function.body
        if isinstance(a,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='config' for z in a.targets)
        or isinstance(a,ast.For) and isinstance(a.target,ast.Name) and a.target.id=='year']
    return payload


def record_training_dependencies(repo,runtime):
    dependencies=training_dependencies(repo);full=code_hash(repo);identity=digest(dependencies)
    commit_bundle(Path(runtime)/'artifacts/model_dependency_manifests'/(full+'-v2'),
                  {'dependencies.json':{'code_hash':full,'training_id':identity,'dependencies':dependencies}},
                  {'code_hash':full,'training_id':identity})
    return identity


def legacy_compatible(runtime,original_code,training_id):
    directory=Path(runtime)/'artifacts/model_dependency_manifests'/(original_code+'-v2')
    if not (directory/'receipt.json').exists():return False
    validate_bundle(directory);record=json.loads((directory/'dependencies.json').read_text())
    return record['code_hash']==original_code and record['training_id']==training_id and digest(record['dependencies'])==training_id


def adopt_bundle(runtime,parent_id,identity,current_code,training_id,expected):
    root=Path(runtime)/'artifacts/auxiliary';parent=root/parent_id;destination=root/identity
    if parent_id==identity:return None
    receipt=validate_bundle(parent);original_code=receipt['metadata']['code_hash']
    if not legacy_compatible(runtime,original_code,training_id):return None
    metric=json.loads((parent/'metrics.json').read_text())
    if metric.get('evidence_kind')!='development' or metric.get('state')!='SUCCEEDED':raise ValueError('Real successful development bundle required')
    if any(metric.get(key)!=value for key,value in expected.items()):raise ValueError('Cache recipe mismatch')
    if (destination/'receipt.json').exists():
        validate_bundle(destination);saved=json.loads((destination/'metrics.json').read_text())
        if any(saved.get(key)!=value for key,value in expected.items()):raise ValueError('Existing cache recipe mismatch')
        return saved
    if not all(name in receipt['artifacts'] for name in ['metrics.json','predictions.json','transform.json','model.bin']):
        raise ValueError('Complete fitted cache bundle required')
    metric={**metric,'run_id':identity,'cache_reuse':True,'cache_parent_run_id':parent_id,'training_implementation_id':training_id}
    files={name:(parent/name).read_bytes() for name in ['predictions.json','transform.json','model.bin']}
    files['metrics.json']=metric
    commit_bundle(destination,files,{'code_hash':original_code,'training_implementation_id':training_id,
                   'data_id':metric['data_id'],'run_id':identity,'cache_materialized_code_hash':current_code,
                   'cache_parent_receipt_id':digest(receipt)})
    return metric
