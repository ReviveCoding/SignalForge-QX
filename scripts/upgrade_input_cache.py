from pathlib import Path
p=Path('src/signalforge/correction_cache.py');text=p.read_text();start=text.index('def corrected_cache(');end=text.index('\ndef adopt_corrected',start)
replacement='''def dataset_proofs(old,ox,folds,x):
    proofs={}
    for year,fold in folds.items():
        cutoff=pd.Timestamp(f'{year}-01-01T00:00Z')-pd.Timedelta(nanoseconds=1)
        test=old[(old.decision_time>=f'{year}-01-01')&(old.decision_time<f'{year+1}-01-01')]
        train=purged_training(old,cutoff,list(zip(test.label_start,test.label_end)))
        dates=sorted(train.decision_time.unique())[-26:]
        if not dates:continue
        valid=train[train.decision_time.isin(dates)]
        inner=purged_training(train,pd.Timestamp(dates[0])-pd.Timedelta(nanoseconds=1),list(zip(valid.label_start,valid.label_end)))
        for partition,previous,current in [('inner_validation',(inner,valid),(fold['inner'],fold['valid'])),('outer_development',(train,test),(fold['train'],fold['test']))]:
            before=partition_signature(*previous,ox);after=partition_signature(*current,x)
            proofs[f'{partition}:{year}']={'original':before,'corrected':after,'exact_match':before==after}
    return proofs


def corrected_cache(repo,runtime,folds,x,data_id):
    datasets={}
    directory=runtime/'artifacts/incidents/eia_2019_duplicate_original_inputs'
    if (directory/'receipt.json').exists():
        receipt=validate_bundle(directory);old=pd.DataFrame(json.loads((directory/'frame.json').read_text()))
        ox=np.array(json.loads((directory/'sequences.json').read_text()));a=json.loads((directory/'acquisition.json').read_text())
        identity=digest({'raw':[r['raw']['sha256'] for r in a['records'] if r['state']=='SUCCEEDED'],'grid':old.decision_time.tolist()})
        if identity!='69892643c84681676c994b74b9acde2002761c60fe566895a668d0e50ff8512c':raise ValueError('Original incident corpus identity mismatch')
        datasets[identity]=[{'proofs':dataset_proofs(old,ox,folds,x),'training_id':None,'receipt_id':digest(receipt)}]
    root=runtime/'artifacts/model_input_snapshots'
    for directory in sorted(root.iterdir()) if root.exists() else []:
        if not (directory/'receipt.json').exists():continue
        receipt=validate_bundle(directory);metadata=receipt['metadata']
        old=pd.DataFrame(json.loads((directory/'frame.json').read_text()));ox=np.array(json.loads((directory/'sequences.json').read_text()))
        a=json.loads((directory/'acquisition.json').read_text())
        from .input_identity import canonical_data_id
        if canonical_data_id(a,old,ox)!=metadata['canonical_data_id']:raise ValueError('Canonical input snapshot identity mismatch')
        source={'proofs':dataset_proofs(old,ox,folds,x),'training_id':metadata['training_id'],'receipt_id':digest(receipt)}
        for identity in {metadata['legacy_data_id'],metadata['canonical_data_id']}:datasets.setdefault(identity,[]).append(source)
    current=numeric_dependencies(training_dependencies(repo));cache={};parent_proofs={}
    root=runtime/'artifacts/auxiliary'
    for parent in sorted(root.iterdir()) if root.exists() else []:
        if not (parent/'receipt.json').exists():continue
        receipt=validate_bundle(parent);metric=json.loads((parent/'metrics.json').read_text())
        if metric.get('state')!='SUCCEEDED' or metric.get('evidence_kind')!='development':continue
        sources=datasets.get(metric.get('data_id'),[])
        if not sources:continue
        dependency=runtime/'artifacts/model_dependency_manifests'/(receipt['metadata']['code_hash']+'-v2')
        if not (dependency/'receipt.json').exists():continue
        validate_bundle(dependency);record=json.loads((dependency/'dependencies.json').read_text())
        if numeric_dependencies(record['dependencies'])!=current:continue
        original_training=metric.get('training_implementation_id',record['training_id'])
        sources=[s for s in sources if s['training_id'] is None or s['training_id']==original_training]
        if len(sources)!=1:continue # Ambiguous legacy input lineage is never guessed.
        source=sources[0];pred=json.loads((parent/'predictions.json').read_text());dates=pd.to_datetime(pred['decision_time'],utc=True)
        for year,fold in folds.items():
            for partition,part in [('inner_validation','valid'),('outer_development','test')]:
                proof=source['proofs'].get(f'{partition}:{year}',{})
                if proof.get('exact_match') and dates.tolist()==pd.to_datetime(fold[part].decision_time,utc=True).tolist():
                    key=digest({'partition':partition,'year':year,'family':metric['family'],'seed':metric['seed'],'trial':metric['trial']})
                    if key not in cache:
                        cache[key]=parent.name;parent_proofs[key]={**proof,'input_receipt':source['receipt_id'],'parent_data_id':metric['data_id']}
    proof_id=digest({'data_id':data_id,'parent_proofs':parent_proofs,'parents':cache,'numeric_dependencies':current})
    commit_bundle(runtime/'artifacts/source_correction_cache_proofs'/proof_id,
                  {'proof.json':{'data_id':data_id,'parent_proofs':parent_proofs,'parents':cache,'numeric_dependencies':current}},
                  {'purpose':'exact_input_stage_reuse','final_access':False})
    return cache,proof_id
'''
text=text[:start]+replacement+text[end:]
text=text.replace("not proof['proofs'][f'{partition}:{year}']['exact_match']", "not proof['parent_proofs'][key]['exact_match']")
p.write_text(text)
