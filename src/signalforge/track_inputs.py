"""Hash-bound Main/Nested development inputs; no reserved or price substitution."""
import json
from pathlib import Path
import pandas as pd
from .runtime import file_hash,digest,atomic_json,now
from .panels import fixed_grid,FeatureSpec,build_features,build_contexts,matched_grid
from .main_labels import build_main_labels


CARD_MAX_BYTES=64*1024**2
PARTITIONED_MAX_BYTES=256*1024**2
PARTITIONED_MAX_PARTS=8

def _input_path(runtime,card):
    if not isinstance(card,dict) or not {'relative_path','sha256'}<=set(card):
        raise ValueError('Canonical input card requires relative_path and sha256')
    path=(Path(runtime)/card['relative_path']).resolve()
    if not path.is_relative_to(Path(runtime).resolve()):raise PermissionError('Input path escapes ext4 runtime')
    if path.stat().st_size>CARD_MAX_BYTES:
        raise RuntimeError('BLOCKED_RESOURCE: bounded canonical input JSON ceiling; use registered partitioned ingestion')
    if file_hash(path)!=card['sha256']:raise ValueError('Input manifest checksum mismatch')
    return path

def read_input_card(runtime,card):
    return json.loads(_input_path(runtime,card).read_text())

def read_partitioned_cards(runtime,cards):
    if not isinstance(cards,list) or not 1<=len(cards)<=PARTITIONED_MAX_PARTS:
        raise ValueError('Bounded nonempty canonical input partitions required')
    rows=[];total=0;seen=set()
    for card in cards:
        path=_input_path(runtime,card)
        identity=(card['relative_path'],card['sha256'])
        if identity in seen:raise ValueError('Duplicate canonical input partition')
        seen.add(identity);total+=path.stat().st_size
        if total>PARTITIONED_MAX_BYTES:
            raise RuntimeError('BLOCKED_RESOURCE: partitioned canonical input total byte ceiling exceeded')
        part=json.loads(path.read_text())
        if not isinstance(part,list):raise ValueError('Canonical input partition must contain a JSON row list')
        rows.extend(part)
    return rows


def prepare_track_inputs(repo,runtime,manifest,track):
    if track not in {'Main-A','Nested-B'}:raise ValueError('Main/Nested track required')
    if manifest.get('schema')!='release_aware_track_inputs_v1':raise ValueError('Registered input schema required')
    if manifest.get('reserved_access') is not False or manifest.get('development_end')!='2023-12-31':
        raise PermissionError('Explicit development-only boundary required')
    if manifest.get('evidence_kind')!='public_data_reconstructed':raise PermissionError('Fixture inputs cannot become public research')
    if manifest.get('qualified_for_final',False) or manifest.get('economic_qualified',False):
        raise PermissionError('Reconstructed track inputs cannot self-qualify final or economics')
    # Validate the separately stored origin calendar before opening target files.
    decisions=read_input_card(runtime,manifest['decisions'])
    grid=fixed_grid(decisions,manifest['universe_card'])
    study=json.loads((repo/'configs/study.json').read_text())
    required=['I0','I1','I2','I3']+(['I4'] if track=='Nested-B' else [])
    if manifest.get('information_sets')!=required:raise ValueError('Complete registered information grid required')
    data=manifest['inputs']
    if 'event_partitions' in data:
        if 'events' in data:raise ValueError('Canonical events must use either one card or registered partitions, not both')
        event_rows=read_partitioned_cards(runtime,data['event_partitions'])
    else:
        event_rows=read_input_card(runtime,data['events'])
    events=pd.DataFrame(event_rows)
    prices=pd.DataFrame(read_input_card(runtime,data['prices']))
    mappings=pd.DataFrame(read_input_card(runtime,data['mappings'])) if 'mappings' in data else None
    registry=read_input_card(runtime,data['evidence_registry']) if 'evidence_registry' in data else None
    specs=[FeatureSpec(**spec) for spec in manifest['features']]
    present=set(events.source)
    needed=set(study['information_sets'][required[-1]])
    if not (needed & present)<={spec.source for spec in specs}:raise ValueError('Every present registered source requires an explicit feature specification')
    mode=manifest.get('price_mode')
    if mode=='P1':
        actions=read_input_card(runtime,data['actions']);coverage=read_input_card(runtime,data['action_coverage'])
        labels=build_main_labels(grid,prices,actions,action_coverage=coverage)
    elif mode=='P0':
        from .p0_labels import build_p0_labels
        labels=build_p0_labels(grid,prices,manifest['p0_basis'])
    else:raise PermissionError('Explicit separate P0/P1 protocol required; no price substitution')
    if not labels.y.notna().any():raise RuntimeError('BLOCKED_DATA: no mature target under the explicit '+mode+' protocol')
    panels={};contexts={};blocked=[]
    for info in required:
        missing=set(study['information_sets'][info])-present
        if missing:
            blocked.append({'information':info,'state':'BLOCKED_DATA','missing_sources':sorted(missing)});continue
        feature=build_features(grid,events,specs,study['information_sets'][info],mappings,strict=False,evidence_registry=registry)
        for column in labels:
            if column not in feature:feature[column]=labels[column]
        panel,x,_=build_contexts(feature)
        panels[info]=panel;contexts[info]=x
    if not panels:raise RuntimeError('BLOCKED_DATA: no registered information branch has its source events')
    identity=digest(manifest)
    return panels,contexts,{'state':'PARTIAL_RECONSTRUCTED_DEVELOPMENT_INPUTS' if blocked else 'READY_RECONSTRUCTED_DEVELOPMENT_INPUTS','track':track,'manifest_id':identity,
        'grid_id':matched_grid(*panels.values()),'grid_rows':len(grid),'mature_targets':int(labels.y.notna().sum()),
        'ready_information_sets':sorted(panels),'blocked_information_sets':blocked,
        'pit_tiers':sorted(events.pit_tier.unique()),'price_mode':mode,'target_contract_id':labels.attrs['target_contract_id'],'qualified_for_final':False,'economic_qualified':False,
        'reserved_access':False,'created_at':now()}


def qualify_track_inputs(repo,runtime,track,version='base'):
    if version not in {'base','v31','v311'}:raise ValueError('Unknown track input version')
    suffix='' if version=='base' else '_'+version
    path=repo/'.local'/('inputs_'+track+suffix+'.json')
    if not path.exists():
        return None,None,{'state':'BLOCKED_DATA','track':track,'manifest_path':str(path),
            'reason':'Hash-bound explicit P0 weekly prices or P1 daily prices/actions, canonical unit/clock sources, outcome-blind universe and as-of mappings absent',
            'qualified_for_final':False,'economic_qualified':False,'reserved_access':False,'created_at':now()}
    return prepare_track_inputs(repo,runtime,json.loads(path.read_text()),track)
