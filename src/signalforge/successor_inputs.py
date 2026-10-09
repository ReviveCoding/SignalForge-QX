"""Hash-bound ext4 panel cache; no source acquisition or scientific transformations."""
import json
import pickle
from pathlib import Path
from .runtime import digest,file_hash,validate_bundle,commit_bundle
from .track_inputs import prepare_track_inputs,read_input_card,read_partitioned_cards


def prepared_track(repo,runtime,track):
    repo=Path(repo);runtime=Path(runtime)
    manifest=json.loads((repo/'.local'/('inputs_'+track+'_v311.json')).read_text())
    if manifest.get('reserved_access') is not False:raise PermissionError('Reserved panel cache forbidden')
    # Validate upstream cards even when cache exists. File paths alone are not identities.
    read_input_card(runtime,manifest['decisions'])
    from .panels import fixed_grid
    fixed_grid(read_input_card(runtime,manifest['decisions']),manifest['universe_card'])
    for name,card in manifest['inputs'].items():
        if name=='event_partitions':read_partitioned_cards(runtime,card)
        else:read_input_card(runtime,card)
    modules=['track_inputs','panels','pit','data','features','labels','p0_labels','calendar','track_neural']
    hashes={name:file_hash(repo/'src/signalforge'/(name+'.py')) for name in modules
            if (repo/'src/signalforge'/(name+'.py')).exists()}
    key=digest({'manifest':manifest,'feature_math':hashes,'study':file_hash(repo/'configs/study.json')})
    directory=runtime/'artifacts/successor_panels'/key
    metadata={'track':track,'panel_key':key,'manifest_id':digest(manifest),'reserved_access':False}
    if (directory/'receipt.json').exists():
        receipt=validate_bundle(directory)
        if receipt['metadata']!=metadata:raise PermissionError('Panel cache identity mismatch')
        # This is an internal, checksum-validated artifact, never a provider pickle.
        panels,contexts,qualification=pickle.loads((directory/'panels.pickle').read_bytes())
    else:
        import fcntl
        lock_path=runtime/'locks'/('successor_panels_'+track+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('RESOURCE_CONTENTION: panel preparation already active') from None
            if (directory/'receipt.json').exists():
                validate_bundle(directory);panels,contexts,qualification=pickle.loads((directory/'panels.pickle').read_bytes())
            else:
                panels,contexts,qualification=prepare_track_inputs(repo,runtime,manifest,track)
                commit_bundle(directory,{'panels.pickle':pickle.dumps((panels,contexts,qualification),protocol=5)},metadata)
    if qualification['manifest_id']!=digest(manifest) or qualification.get('blocked_information_sets'):
        raise PermissionError('INTEGRITY_OR_HASH_GATE: panel qualification differs')
    from .track_engine import validate_panel
    for info,panel in panels.items():validate_panel(panel,contexts[info])
    return panels,contexts,qualification
