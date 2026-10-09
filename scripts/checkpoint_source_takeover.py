"""Evidence-backed in-place takeover checkpoint, including independently blocked branches."""
import json
from pathlib import Path
import pandas as pd
from signalforge.runtime import paths,now,code_hash,file_hash,digest,atomic_json,commit_bundle,validate_bundle
from signalforge.source_requests import keyed_pilot
from signalforge.macro_integration import build_macro_integration
from signalforge.track_source_integration import build_extension_v311,_load_source_events
from signalforge.completion import current_full_validation,successor_boundary
from signalforge.nport_reclock import conservative_development_clock

repo,runtime=paths();stages=[]
for name,operation in [('fred',lambda:keyed_pilot(repo,runtime,'fred')),
                       ('macro',lambda:build_macro_integration(repo,runtime)),
                       ('source_extension',lambda:build_extension_v311(repo,runtime))]:
    try:
        result=operation();stages.append({'branch':name,'state':result['state']})
    except (PermissionError,RuntimeError,ValueError,FileNotFoundError) as error:
        stages.append({'branch':name,'state':'BLOCKED_AUTH' if isinstance(error,PermissionError) else 'BLOCKED_DATA',
                       'error_type':type(error).__name__,'reason':str(error)})

sources={}
for source in ['cftc','eia','nport']:
    rows,report=_load_source_events(repo,runtime,source)
    if rows is None:raise ValueError('Existing canonical source no longer reusable: '+source)
    validate_bundle((runtime/report['events_relative_path']).parent)
    sources[source]={'state':report['state'],'events':len(rows),'rows':report.get('rows'),
                     'events_sha256':report['events_sha256'],'pit_tier':report['pit_tier'],
                     'qualified_for_final':report['qualified_for_final']}
    if source=='nport':
        parent=json.loads((repo/'reports/nport_bulk_integration.json').read_text())
        folder=(runtime/parent['events_relative_path']).parent;validate_bundle(folder)
        clocks={x['ACCESSION_NUMBER']:pd.Timestamp(x['accepted_at']) for x in json.loads((folder/'clocks.json').read_text())}
        originals={tuple(x[k] for k in ['entity','field','reference_time','accession']):x for x in json.loads((folder/'events.json').read_text())}
        for row in rows:
            expected=conservative_development_clock(clocks[row['accession']])
            if pd.Timestamp(row['available_at'])!=expected:raise ValueError('NPORT conservative clock mismatch')
            old=originals[tuple(row[k] for k in ['entity','field','reference_time','accession'])]
            if any(row[k]!=old[k] for k in ['value','unit','raw_hash']):raise ValueError('NPORT re-clock changed source values')
        sources[source].update(reclock_verified_events=len(rows),blocked=report.get('blocked',[]))

base_path=repo/'reports/track_input_qualification_base.json'
base=json.loads(base_path.read_text())
base_results={}
for row in base['tracks']:
    cpu=row['cpu_development'];success=[x for x in cpu['results'] if x['state']=='SUCCEEDED']
    for fit in success:
        validate_bundle(runtime/'artifacts/track_development'/fit['run_id'])
        if fit.get('model_id'):validate_bundle(runtime/'artifacts/track_models'/fit['model_id'])
    base_results[row['track']]={'state':cpu['state'],'successful_outer_results':len(success),
                               'blocked_folds':cpu['blocked_folds'],'ready_information_sets':row['ready_information_sets']}

audit=json.loads((repo/'reports/fred_cached_page_audit.json').read_text())
versions=sum(x.get('observation_versions',0) for x in audit['pages'])
result={'state':'BLOCKED_PROCESS_ONLY_FRED_CREDENTIAL_AFTER_AVAILABLE_WORK','created_at':now(),
        'source_tree_hash':code_hash(repo),'branches':stages,'source_reuse':sources,'base_cpu_reuse':base_results,
        'fred_cached_registered_pages':len(audit['pages']),'fred_cached_decoded_observation_versions':versions,
        'fred_acquisition_complete':False,'macro_event_counts':None,
        'v311_manifest_present':{t:(repo/'.local'/f'inputs_{t}_v311.json').exists() for t in ['Main-A','Nested-B']},
        'v311_qualification_receipt':'reports/track_input_qualification_v311.json',
        'validation':current_full_validation(repo),'successor_gpu_boundary':successor_boundary(repo),
        'plan_hashes':{n:file_hash(repo/'.local'/n) for n in ['fred_request_plan.json','fred_request_plan_v31.json']},
        'freeze_receipt_present':(repo/'.local/freeze_receipt.json').exists(),'reserved_access':False,
        'resume_command':'pwsh -NoProfile -File .\\scripts\\Resume-SignalForge-Macro.ps1',
        'claim_boundary':'Real cached reconstructed development evidence and software validation; no completed FRED acquisition, source-extended fitting, strict PIT, P1 or final claim.'}
identity=digest(result)
commit_bundle(runtime/'artifacts/takeover_development_checkpoints'/identity,{'checkpoint.json':result},{'reserved_access':False})
result['artifact_id']=identity;atomic_json(repo/'reports/takeover_development_checkpoint.json',result)
print(json.dumps(result,indent=2))
