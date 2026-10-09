"""Scientific access boundary. Authorization cannot substitute for verified evidence."""
import json
from pathlib import Path
from .runtime import digest, file_hash, atomic_json, now, code_hash

REQUIRED_FREEZE_COMPONENTS = {'source_snapshot','features','transforms','labels_splits','models',
                              'ensemble','calibration','eligibility_fallback','metrics_slices',
                              'contrast_family','hpo_budget','environment','code','controller','P10_predictive','P11'}
REQUIRED_FREEZE_GATES={'data','pit','baselines','proposed','development','postprocessing','systems','integrity'}


def protocol_hash(repo):
    return digest({str(p.relative_to(repo)):file_hash(p) for p in sorted((Path(repo)/'configs').glob('*.json'))})


def verify_qualification(receipt, repo):
    """Validate substantive qualification separately from artifact integrity.

    Diagnostic/synthetic runs cannot authorize final research. Economics remains
    a separate branch, so a missing P1 panel need not block predictive receipts.
    """
    gates=receipt.get('gates',{})
    if not REQUIRED_FREEZE_GATES<=set(gates):
        raise PermissionError('Scientific qualification gates incomplete')
    for name in REQUIRED_FREEZE_GATES:
        gate=gates[name]
        if gate.get('state')!='SUCCEEDED' or gate.get('evidence_kind')!='qualified_real_development':
            raise PermissionError('Unqualified scientific gate: '+name)
        if not gate.get('evidence_components') or not set(gate['evidence_components'])<=set(receipt['components']):
            raise PermissionError('Gate lacks frozen evidence: '+name)
    if receipt.get('protocol_hash')!=protocol_hash(repo) or receipt.get('source_code_hash')!=code_hash(repo):
        raise PermissionError('Current protocol or source tree differs from freeze')
    if receipt.get('economics_state') not in {'QUALIFIED_PRICE_P1','BLOCKED_PRICE_P1'}:
        raise PermissionError('Economics branch must have an explicit separate qualification state')
    if not receipt.get('registered_candidate_ids') or not receipt.get('registered_contrast_ids'):
        raise PermissionError('Complete frozen candidate and comparison families required')
    if len(set(receipt['registered_candidate_ids']))!=len(receipt['registered_candidate_ids']):
        raise PermissionError('Duplicate frozen candidates')
    study_path=Path(repo)/'configs/study.json'
    if not study_path.exists():raise PermissionError('Actual registered study contract required')
    expected=json.loads(study_path.read_text())['tracks']
    if set(receipt.get('tracks',{}))!=set(expected):raise PermissionError('All track qualification receipts required')
    for track,qualification in receipt['tracks'].items():
        if qualification.get('prediction_state')!='QUALIFIED' or qualification.get('strict_PIT_qualified') is not True:
            raise PermissionError('Track predictive/strict-PIT gate incomplete: '+track)
    # Qualified flags and file hashes cannot substitute for canonical source evidence.
    source=json.loads(Path(receipt['components']['source_snapshot']['path']).read_text())
    if set(source.get('tracks',{}))!=set(expected):raise PermissionError('All-track original source snapshots required')
    import pandas as pd
    from .data import validate_events
    for track,card in source['tracks'].items():
        events=pd.DataFrame(card.get('events',[]))
        if events.empty or 'pit_tier' not in events or not events.pit_tier.eq('A').all():
            raise PermissionError('Strict original source events required: '+track)
        try:validate_events(events,card.get('evidence_registry'))
        except (ValueError,KeyError,FileNotFoundError) as error:raise PermissionError('Unverified original source snapshot: '+track) from error
        if pd.to_datetime(events.available_at,utc=True).ge('2024-01-01T00:00Z').any():raise PermissionError('Reserved source payload in development freeze')
    environment=Path(repo)/'reports/environment_audit.json'
    if not environment.exists() or file_hash(environment)!=receipt['components']['environment']['sha256']:
        raise PermissionError('Current qualified environment receipt differs from freeze')


def verify_freeze(receipt, repo, runtime, auth):
    if (auth.get('authorized') is not True or auth.get('study_id')!='sgqx-v3' or
        auth.get('scope')!='one_registered_frozen_batch_after_all_track_gates'):
        raise PermissionError('Scientific final authorization absent or scope mismatch')
    if receipt.get('study_id')!='sgqx-v3' or receipt.get('status')!='READY_FOR_FINAL':
        raise PermissionError('Valid READY_FOR_FINAL receipt required')
    components=receipt.get('components',{})
    if not REQUIRED_FREEZE_COMPONENTS <= set(components) or not receipt.get('protocol_hash'):
        raise PermissionError('Incomplete freeze components')
    roots=[Path(repo).resolve(),Path(runtime).resolve()]
    for name,artifact in components.items():
        path=Path(artifact['path']).resolve()
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
            raise PermissionError('Freeze artifact outside study or missing: '+name)
        if file_hash(path)!=artifact['sha256']:
            raise PermissionError('Freeze artifact changed: '+name)
    verify_qualification(receipt,repo)
    verify_frozen_models(receipt,runtime)
    return digest(receipt)


def verify_frozen_models(receipt,runtime):
    components=receipt['components']
    model_manifest=json.loads(Path(components['models']['path']).read_text())
    candidates=model_manifest.get('candidates',[])
    if {c.get('candidate_id') for c in candidates}!=set(receipt['registered_candidate_ids']):raise PermissionError('Actual full frozen model family required')
    from .runtime import validate_bundle
    source_id=digest(json.loads(Path(components['source_snapshot']['path']).read_text()))
    for candidate in candidates:
        if candidate.get('track') not in receipt['tracks'] or not candidate.get('members'):raise PermissionError('Frozen track/model members required')
        for member in candidate['members']:
            bundle=(Path(runtime)/member['relative_bundle']).resolve()
            if not bundle.is_relative_to(Path(runtime).resolve()):raise PermissionError('Frozen model outside ext4 runtime')
            model_receipt=validate_bundle(bundle)
            if digest(model_receipt)!=member['receipt_id']:raise PermissionError('Actual model receipt mismatch')
            metadata=model_receipt['metadata']
            if metadata.get('source_snapshot_id')!=source_id or metadata.get('evidence_kind')!='qualified_real_development':
                raise PermissionError('Model training lacks strict original-source lineage')
            cutoff=pd_timestamp(metadata.get('max_training_label_available_at'))
            if cutoff is None or cutoff>=pd_timestamp('2023-01-01T00:00Z'):raise PermissionError('Frozen member training maturity exceeds registered cutoff')
    return True


def pd_timestamp(value):
    import pandas as pd
    if value is None:return None
    try:result=pd.Timestamp(value)
    except (ValueError,TypeError) as error:raise PermissionError('Invalid frozen maturity clock') from error
    return result if result.tzinfo is not None and not pd.isna(result) else None


def create_freeze(repo,runtime):
    """Build a reviewable receipt from actual completed qualification manifests."""
    proposal=Path(repo)/'reports/freeze_candidate_manifest.json'
    if not proposal.exists():raise RuntimeError('BLOCKED_DATA: completed selection/calibration/qualification candidate manifest absent')
    content=json.loads(proposal.read_text());components={}
    for name,path in content['artifact_paths'].items():
        artifact=Path(path).resolve()
        if not any(artifact.is_relative_to(root.resolve()) for root in [Path(repo),Path(runtime)]) or not artifact.is_file():
            raise PermissionError('Missing/out-of-scope freeze component: '+name)
        components[name]={'path':str(artifact),'sha256':file_hash(artifact)}
    if not REQUIRED_FREEZE_COMPONENTS<=set(components):raise PermissionError('Incomplete frozen artifacts')
    receipt={**{k:v for k,v in content.items() if k!='artifact_paths'},'study_id':'sgqx-v3','status':'READY_FOR_FINAL',
             'components':components,'protocol_hash':protocol_hash(repo),'source_code_hash':code_hash(repo),'frozen_at':now()}
    verify_qualification(receipt,repo)
    verify_frozen_models(receipt,runtime)
    # Freeze can occur without final authorization. No reserved data is accessed.
    directory=Path(runtime)/'artifacts/freezes'/digest(receipt)
    from .runtime import commit_bundle
    commit_bundle(directory,{'freeze_receipt.json':receipt},{'study_id':'sgqx-v3'})
    atomic_json(Path(repo)/'.local/freeze_receipt.json',receipt)
    return receipt


def authorize_final(receipt,repo,runtime,auth,mode):
    if mode!='locked':
        raise PermissionError('Development cannot access reserved labels')
    freeze_id=verify_freeze(receipt,repo,runtime,auth)
    path=Path(runtime)/'ledger/final_access.json'
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/'final_access.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['freeze_id']!=freeze_id:
                raise PermissionError('One frozen batch already consumed; new candidate denied')
        else:
            from .contamination import history_records,assert_unused_period
            assert_unused_period(history_records(Path(runtime)),'2024-01-01T00:00Z','2026-07-01T00:00Z')
            atomic_json(path,{'freeze_id':freeze_id,'first_authorized_at':now(),'study_id':'sgqx-v3',
                             'candidate_ids':receipt['registered_candidate_ids']})
    return freeze_id


def forward_clock(frozen_at,decision_time,outcome_at,actual_now):
    from datetime import datetime
    times=[datetime.fromisoformat(t.replace('Z','+00:00')) for t in [frozen_at,decision_time,outcome_at,actual_now]]
    if any(t.tzinfo is None for t in times):
        raise ValueError('Aware timestamps required')
    freeze,decision,outcome,current=times
    if not freeze<decision<=current<outcome:
        raise PermissionError('Missed/late or historical replay cannot be prospective')
    return True
