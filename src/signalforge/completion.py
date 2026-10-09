"""Successor completion protocol guards; never mutates the parent v3 budgets or evidence."""
import json
from pathlib import Path
from .runtime import file_hash, digest, code_hash

PARENT_PILOT_CEILING=120
PARENT_GPU_SECONDS=43200


def current_full_validation(repo):
    """GPU admission requires all four suites and the exact current qualification."""
    repo=Path(repo);path=repo/'reports/test_execution.json'
    result={'present':path.exists(),'passed':False,'current_tree':False,'tests':0,
            'qualification_content_current':False}
    if not path.exists():return result
    doc=json.loads(path.read_text());rows=doc.get('results',[])
    result['tests']=sum(int(r.get('counts',{}).get('tests',0) or 0) for r in rows)
    result['current_tree']=bool(rows) and {r.get('source_tree_hash') for r in rows}=={code_hash(repo)}
    qpath=repo/'reports/track_input_qualification_v311.json'
    expected=file_hash(qpath) if qpath.exists() else None
    result['qualification_content_current']=doc.get('qualification_sha256')==expected
    newer=qpath.exists() and qpath.stat().st_mtime_ns>path.stat().st_mtime_ns
    complete={r.get('suite') for r in rows}=={'handoff','unit','gpu','real_source'} and len(rows)==4
    result['passed']=bool(complete and result['current_tree'] and not newer and
        result['qualification_content_current'] and doc.get('inputs_unchanged') is True and
        all(r.get('exit_code')==0 and r.get('counts',{}).get('tests',0)>0 and
            not any(r.get('counts',{}).get(k,0) for k in ['failures','errors','skipped']) for r in rows))
    return result


def load_protocol(repo):
    repo=Path(repo);path=repo/'configs/completion_extension_v31.json'
    data=json.loads(path.read_text())
    if data.get('protocol_id')!='sgqx-v3.1-track-gpu' or data.get('parent_study')!='sgqx-v3':
        raise ValueError('Completion successor identity mismatch')
    parent=data.get('parent_budget',{})
    if parent.get('pilot_completed_fit_ceiling')!=PARENT_PILOT_CEILING or parent.get('development_gpu_seconds')!=PARENT_GPU_SECONDS or parent.get('reset_permitted') is not False:
        raise PermissionError('Parent v3 budget contract changed')
    local=json.loads((repo/'configs/local_rtx4090_laptop.json').read_text())['budget']
    if local.get('pilot_completed_fit_ceiling')!=PARENT_PILOT_CEILING or local.get('development_gpu_seconds')!=PARENT_GPU_SECONDS:
        raise PermissionError('Live parent budget differs from immutable completion contract')
    successor=data.get('successor_gpu',{})
    if successor.get('parent_ledger_reuse') is not False or successor.get('physical_gpus')!=1:
        raise PermissionError('Successor requires a separate one-GPU ledger')
    if not (0<successor.get('pilot_fit_ceiling',0)<=60 and 0<successor.get('pilot_gpu_seconds',0)<=3600 and 0<successor.get('full_gpu_seconds',0)<=21600):
        raise ValueError('Successor budget outside preregistered ceiling')
    return data


def _latest_track_qualification(repo):
    repo=Path(repo)
    preferred=repo/'reports/track_input_qualification_v311.json'
    fallback=repo/'reports/track_input_qualification.json'
    path=preferred if preferred.exists() else fallback
    return path,(json.loads(path.read_text()) if path.exists() else None)


def successor_track_boundary(repo,track):
    repo=Path(repo);protocol=load_protocol(repo)
    if track not in {'Main-A','Nested-B'}:raise ValueError('Unknown successor track')
    capacity=repo/'reports/auxiliary_capacity_bill.json'
    if capacity.exists():
        used=json.loads(capacity.read_text()).get('cumulative_pilot_completed')
        if used!=PARENT_PILOT_CEILING:raise ValueError('Parent cumulative pilot evidence must remain 120/120')
    manifest=repo/'.local'/('inputs_'+track+'_v311.json')
    if not manifest.exists():
        return {'track':track,'state':'BLOCKED_DATA','protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
            'reason':'Versioned source-extension manifest absent for '+track,'successor_gpu_started':False}
    qpath,qualification=_latest_track_qualification(repo)
    if qualification is None:
        return {'track':track,'state':'BLOCKED_DATA','protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
            'reason':'Track input qualification receipt absent','successor_gpu_started':False}
    rows={row.get('track'):row for row in qualification.get('tracks',[])}
    row=rows.get(track)
    if row is None:
        return {'track':track,'state':'BLOCKED_DATA','protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
            'reason':'Track CPU qualification row absent for '+track,'successor_gpu_started':False}
    cpu=row.get('cpu_development',{})
    if cpu.get('state')!='SUCCEEDED_DEVELOPMENT_SOFTWARE' or cpu.get('blocked_folds'):
        return {'track':track,'state':'BLOCKED_DATA','protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
            'reason':'Successful fixed-grid CPU development must precede successor GPU admission: '+track,
            'successor_gpu_started':False}
    required={'I0','I1','I2','I3'}|({'I4'} if track=='Nested-B' else set())
    ready=set(row.get('ready_information_sets',[]))
    blocked={x.get('information') for x in row.get('blocked_information_sets',[])}
    observed={r.get('information') for r in cpu.get('results',[]) if r.get('state')=='SUCCEEDED'}
    if required & blocked or not required<=ready:
        missing=sorted(required-ready)
        return {'track':track,'state':'BLOCKED_DATA','protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
            'reason':'Registered information sets remain source-blocked for '+track+': '+','.join(missing or sorted(required&blocked)),
            'successor_gpu_started':False}
    if not required<=observed:
        return {'track':track,'state':'BLOCKED_DATA','protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
            'reason':'CPU development does not cover every required information set for '+track,
            'successor_gpu_started':False}
    return {'track':track,'state':'READY_FOR_MEASURED_SUCCESSOR_PILOT','protocol_id':protocol['protocol_id'],
        'parent_budget_unchanged':True,'successor_gpu_started':False,'qualification_receipt':str(qpath.relative_to(repo)),
        'protocol_sha256':file_hash(repo/'configs/completion_extension_v31.json')}


def successor_boundary(repo):
    repo=Path(repo);protocol=load_protocol(repo)
    tracks={track:successor_track_boundary(repo,track) for track in ['Main-A','Nested-B']}
    ready=[track for track,state in tracks.items() if state['state']=='READY_FOR_MEASURED_SUCCESSOR_PILOT']
    state='READY_FOR_MEASURED_SUCCESSOR_PILOT' if len(ready)==2 else 'PARTIAL_TRACK_READY' if ready else 'BLOCKED_DATA'
    return {'state':state,'protocol_id':protocol['protocol_id'],'parent_budget_unchanged':True,
        'ready_tracks':ready,'track_boundaries':tracks,'successor_gpu_started':False,
        'protocol_sha256':file_hash(repo/'configs/completion_extension_v31.json')}


def calibration_boundary(repo,track):
    repo=Path(repo)
    if track not in {'Auxiliary-C','Main-A','Nested-B'}:raise ValueError('Unknown calibration track')
    minimum=json.loads((repo/'configs/statistical_contract.json').read_text())['calibration']['minimum_distinct_dates']
    if track=='Auxiliary-C':
        path=repo/'reports/eia_2023_calibration_source_audit.json'
        if not path.exists():return {'track':track,'state':'BLOCKED_SOURCE_CONSISTENCY','qualified_for_freeze':False,'minimum_distinct_dates':minimum}
        audit=json.loads(path.read_text())
        return {'track':track,'state':audit.get('state','BLOCKED_SOURCE_CONSISTENCY'),'qualified_for_freeze':False,
            'minimum_distinct_dates':minimum,'evidence_sha256':file_hash(path)}
    path=repo/'reports'/('calibration_'+track.replace('-','_')+'.json')
    if not path.exists():
        return {'track':track,'state':'NOT_FIT','qualified_for_freeze':False,'minimum_distinct_dates':minimum,
            'auxiliary_block_independent':True}
    receipt=json.loads(path.read_text())
    if receipt.get('track')!=track:raise ValueError('Calibration receipt track mismatch')
    return {'track':track,'state':receipt.get('state','INVALID'),'qualified_for_freeze':bool(receipt.get('qualified_for_freeze',False)),
        'minimum_distinct_dates':minimum,'auxiliary_block_independent':True,'receipt_id':digest(receipt)}
