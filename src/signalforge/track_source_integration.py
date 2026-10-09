"""Versioned source extensions for Main-A/Nested-B without mutating base I0 manifests."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import file_hash,digest,commit_bundle,atomic_json,now
from .data import validate_events
from .track_inputs import read_input_card,prepare_track_inputs

REPORTS={
    'cftc':('reports/cftc_canonical_integration.json','SUCCEEDED_RECONSTRUCTED_CFTC_POSITIONING'),
    'eia':('reports/eia_market_integration.json','SUCCEEDED_RECONSTRUCTED_EIA_USO_ACTIVITY'),
    'macro':('reports/macro_canonical_integration.json','SUCCEEDED_RECONSTRUCTED_MACRO_VINTAGES'),
    'nport':('reports/nport_bulk_integration.json','SUCCEEDED_RECONSTRUCTED_NPORT_TARGET_FLOWS'),
}


def _load_source_events(repo,runtime,source):
    rel,state=REPORTS[source]
    if source=='nport' and (repo/'reports/nport_bulk_integration_v311.json').exists():
        rel='reports/nport_bulk_integration_v311.json'
    rp=repo/rel
    if not rp.exists():return None,None
    report=json.loads(rp.read_text())
    if report.get('state')!=state:return None,report
    path=(runtime/report['events_relative_path']).resolve()
    if not path.is_relative_to(runtime.resolve()) or file_hash(path)!=report['events_sha256']:
        raise ValueError(source+' canonical event receipt mismatch')
    rows=json.loads(path.read_text())
    frame=pd.DataFrame(rows)
    if len(frame):
        if not frame.source.eq(source).all():raise ValueError(source+' canonical source label mismatch')
        validate_events(frame)
    return rows,report


def _card(runtime,folder,name):
    path=folder/name
    return {'relative_path':str(path.relative_to(runtime)),'sha256':file_hash(path)}


def _i0_signature(repo,runtime,manifest,track):
    panels,contexts,q=prepare_track_inputs(repo,runtime,manifest,track)
    panel=panels['I0'];x=contexts['I0']
    cols=['decision_time','asset','y','label_start','label_end','label_available_at','max_dependency_available_at',
          'eligible','context_eligible','price_mode','p0_origin_mark_available_at']
    cols=[c for c in cols if c in panel]
    body=panel[cols].copy()
    for c in body:
        if 'time' in c or c in {'label_start','label_end','label_available_at','max_dependency_available_at','p0_origin_mark_available_at'}:
            body[c]=body[c].astype(str)
    panel_records=json.loads(body.to_json(orient='records',date_format='iso'))
    a=np.asarray(x,dtype=np.float64)
    numeric=np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0).astype('<f8',copy=False)
    mask=np.isfinite(a)
    array_hash=hashlib.sha256(numeric.tobytes(order='C')+np.packbits(mask.reshape(-1)).tobytes()).hexdigest()
    return digest({'panel':panel_records,'x_shape':list(a.shape),'x_hash':array_hash,
                   'feature_names':panel.attrs.get('feature_names',[]),'target_contract_id':q['target_contract_id'],'grid_id':q['grid_id']})


def _build(repo,runtime,spec_name,suffix,report_name,protocol_id):
    repo,runtime=Path(repo),Path(runtime)
    spec_path=repo/'configs'/spec_name
    spec_cfg=json.loads(spec_path.read_text())
    if spec_cfg.get('outcome_blind') is not True:raise PermissionError('Outcome-blind feature contract required')
    available={};source_rows=[]
    for source in REPORTS:
        rows,report=_load_source_events(repo,runtime,source)
        if rows is not None:
            available[source]=report;source_rows.extend(rows)
    if 'macro' not in available:
        raise RuntimeError('BLOCKED_DATA: canonical macro integration absent')
    # Main-A can be source-complete without N-PORT; Nested-B I4 remains independently blocked until N-PORT exists.
    outputs=[]
    for track in ['Main-A','Nested-B']:
        base_path=repo/'.local'/('inputs_'+track+'.json')
        if not base_path.exists():raise RuntimeError('BLOCKED_DATA: base Tiingo manifest absent for '+track)
        base=json.loads(base_path.read_text())
        market_rows=read_input_card(runtime,base['inputs']['events'])
        combined=market_rows+source_rows
        frame=pd.DataFrame(combined)
        validate_events(frame)
        keys=['entity','source','field','reference_time','available_at']
        if frame.duplicated(keys).any():raise ValueError('Duplicate canonical source event after extension')
        source_hashes={s:r['events_sha256'] for s,r in available.items()}
        # Reuse immutable per-source canonical cards instead of copying all events
        # into one >64 MiB JSON.  Each partition stays under the existing card
        # ceiling; the manifest binds their exact ordered hashes.
        event_partitions=[json.loads(json.dumps(base['inputs']['events']))]
        partition_sources=['market']
        for source in sorted(available):
            report=available[source]
            event_partitions.append({'relative_path':report['events_relative_path'],'sha256':report['events_sha256']})
            partition_sources.append(source)
        identity=digest({'base_manifest':digest(base),'feature_contract':file_hash(spec_path),
                         'source_hashes':source_hashes,'event_partitions':event_partitions,
                         'partition_sources':partition_sources,'protocol_id':protocol_id})
        folder=runtime/'artifacts'/('track_source_inputs_'+suffix)/identity
        partition_index={'schema':'canonical_event_partitions_v1','event_partitions':event_partitions,
                         'partition_sources':partition_sources,'source_hashes':source_hashes,
                         'total_event_rows':len(frame),'reserved_access':False}
        commit_bundle(folder,{'partition_index.json':partition_index},{'track':track,'protocol_id':protocol_id,
            'qualified_for_final':False,'reserved_access':False})
        manifest=json.loads(json.dumps(base))
        manifest['inputs'].pop('events',None)
        manifest['inputs']['event_partitions']=event_partitions
        manifest['features']=base['features']+sum((spec_cfg['sources'][s] for s in ['cftc','eia','macro','nport'] if s in available),[])
        manifest['extension']={'protocol_id':protocol_id,'base_manifest_sha256':file_hash(base_path),
            'source_event_hashes':source_hashes,'feature_contract_sha256':file_hash(spec_path),
            'feature_contract':str(spec_path.relative_to(repo)),
            'partition_index':_card(runtime,folder,'partition_index.json'),
            'partition_sources':partition_sources,'outcome_blind':True,'qualified_for_final':False}
        successor_path=repo/'.local'/('inputs_'+track+'_'+suffix+'.json')
        base_sig=_i0_signature(repo,runtime,base,track)
        successor_sig=_i0_signature(repo,runtime,manifest,track)
        if base_sig!=successor_sig:
            raise ValueError('I0 parity changed under source extension for '+track)
        _,_,qualification=prepare_track_inputs(repo,runtime,manifest,track)
        atomic_json(successor_path,manifest)
        outputs.append({'track':track,'state':'READY_VERSIONED_SOURCE_EXTENSION','manifest_path':str(successor_path),
            'manifest_id':digest(manifest),'base_i0_signature':base_sig,'successor_i0_signature':successor_sig,
            'available_sources':sorted(available),'source_event_rows':len(source_rows),
            'ready_information_sets':qualification['ready_information_sets'],
            'blocked_information_sets':qualification['blocked_information_sets']})
    result={'state':'READY_VERSIONED_TRACK_SOURCE_INPUTS','protocol_id':protocol_id,'created_at':now(),'tracks':outputs,
        'available_sources':sorted(available),'feature_contract':str(spec_path.relative_to(repo)),
        'feature_contract_sha256':file_hash(spec_path),'reserved_access':False,'qualified_for_final':False,'economic_qualified':False}
    atomic_json(repo/'reports'/report_name,result)
    return result


def build_extension(repo,runtime):
    return _build(repo,runtime,'track_feature_specs_v31.json','v31','track_source_input_extension_v31.json','sgqx-v3.1-track-gpu')


def build_extension_v311(repo,runtime):
    return _build(repo,runtime,'track_feature_specs_v311.json','v311','track_source_input_extension_v311.json','sgqx-v3.1.1-track-sources')
