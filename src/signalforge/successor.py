"""Outcome-blind successor authorization, frozen plans and measured-cost admission."""
import json
import math
import sqlite3
from pathlib import Path
from .runtime import digest, file_hash, atomic_json, now, validate_bundle, code_hash
from .completion import load_protocol

FAMILIES=['lightgbm','xgboost','mlp','gru','rgmf_linear','rgmf_gru']
FIXED={family:(1.0 if family in {'lightgbm','xgboost'} else [16,0.001]) for family in FAMILIES}


def successor_authorized(repo):
    repo=Path(repo);path=repo/'.local/ALLOW_SUCCESSOR_GPU'
    if not path.exists():return False
    auth=json.loads(path.read_text(encoding='utf-8-sig'))
    expected={'authorized':True,'scope':'registered_successor_gpu_development_only',
              'protocol_id':'sgqx-v3.1-track-gpu',
              'protocol_sha256':file_hash(repo/'configs/completion_extension_v31.json'),
              'reserved_access':False,'budget_increase_authorized':False,
              'pilot_fit_ceiling':60,'pilot_gpu_seconds':3600,'full_gpu_seconds':21600}
    if any(auth.get(k)!=v for k,v in expected.items()):
        raise PermissionError('AUTH_GATE: invalid successor development authorization')
    load_protocol(repo)
    return True


def require_successor_access(repo):
    repo=Path(repo)
    if not successor_authorized(repo):raise PermissionError('AUTH_GATE: successor development authorization absent')
    for name in ['PAUSE_BEFORE_GPU','STOP_GPU_WATCHER','STOP_SUCCESSOR_GPU_RECOVERY']:
        if (repo/'.local'/name).exists():raise PermissionError('AUTH_GATE: '+name+' control is armed')


def ledger_snapshot(path,ceiling=None):
    path=Path(path)
    if not path.exists():return {'present':False,'entries':0,'charged_seconds':0.,'charges':[]}
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        stored=db.execute('SELECT ceiling FROM budget_settings WHERE id=1').fetchone()[0]
        rows=db.execute('SELECT id,seconds,state FROM compute_charges ORDER BY id').fetchall()
    if ceiling is not None and stored!=ceiling:raise PermissionError('RESOURCE_OR_BUDGET_GATE: immutable ceiling mismatch')
    return {'present':True,'ceiling_seconds':stored,'entries':len(rows),
            'charged_seconds':sum(r[1] for r in rows),'charges_digest':digest(rows),
            'charges':[{'id':r[0],'seconds':r[1],'state':r[2]} for r in rows]}


def assert_parent_unchanged(runtime,expected):
    actual=ledger_snapshot(Path(runtime)/'ledger/development_compute.sqlite',43200)
    for k in ['entries','charges_digest']:
        if actual[k]!=expected[k]:raise PermissionError('INTEGRITY_OR_HASH_GATE: parent ledger changed')
    return actual


def execution_plan(repo):
    """Derive both registered-grid and explicitly bounded fixed-setting designs.

    The fixed-setting fallback is a separate development comparison, never a
    successful tuned-study substitution. Both designs are frozen before pilots.
    """
    repo=Path(repo);protocol=load_protocol(repo);study=json.loads((repo/'configs/study.json').read_text())
    seeds=protocol['successor_gpu']['seeds']
    if seeds!=[11,37,71] or study['hpo']['development_seeds']!=seeds:raise ValueError('Registered seeds differ')
    tracks={t:{'information_sets':list(protocol['information_sets'][t]),
               'outer_years':study['splits']['track_folds'][t]['outer_years']} for t in ['Main-A','Nested-B']}
    blocks=sum(len(t['information_sets'])*len(t['outer_years']) for t in tracks.values())
    return {'protocol_id':protocol['protocol_id'],'plan_version':'successor-full-v1',
            'outcome_blind':True,'reserved_access':False,'qualified_for_final':False,
            'families':FAMILIES,'conditional_families':protocol['successor_gpu']['conditional_families'],
            'conditional_gate':protocol['successor_gpu']['conditional_gate'],'seeds':seeds,'tracks':tracks,
            'conditional_gate_scope':'every registered mature training fold; pooled rows never count as dates',
            'plans':{
                'registered_hpo':{'mode':'registered_hpo','trials':study['hpo']['full_trials_per_family_fold'],
                                  'neural_widths':[8,16,32,48,64],'learning_rates':[0.0003,0.001,0.003,0.01],
                                  'tree_regularization':'geomspace(0.0001,10000,20)',
                                  'planned_fit_count':blocks*len(FAMILIES)*(study['hpo']['full_trials_per_family_fold']+len(seeds))},
                'fixed_setting_development':{'mode':'fixed_setting_development','fixed_settings':FIXED,
                                  'planned_fit_count':blocks*len(FAMILIES)*len(seeds),
                                  'claim_boundary':'fixed-setting development comparisons; no tuned-family superiority claim'}},
            'selection_rule':'first cost-admissible plan in frozen order; never predictive performance',
            'cost_rule':{'margin_multiplier':2.0,'maximum_width_compute_multiplier':16.0,
                         'gate':'all pilot families compatible; conservative sum <= remaining 21600 seconds'},
            'full_gpu_seconds':21600,'full_ledger':'ledger/successor_gpu_full_compute.sqlite',
            'pilot_ledger':'ledger/successor_gpu_pilot_compute.sqlite',
            'parent_ledger_reuse':False,
            'config_hashes':{str(p.relative_to(repo)):file_hash(p) for p in
                [repo/'configs/completion_extension_v31.json',repo/'configs/study.json',repo/'configs/statistical_contract.json']}}


def freeze_execution_plan(repo,runtime):
    repo=Path(repo);runtime=Path(runtime);plan=execution_plan(repo);identity=digest(plan)
    config=repo/'configs/successor_full_study_v31.json'
    if not config.exists() or json.loads(config.read_text())!=plan:
        raise PermissionError('Frozen full-study config absent or differs from registered derivation')
    from .runtime import commit_bundle
    bundle=runtime/'artifacts/successor_execution_plans'/identity
    receipt=commit_bundle(bundle,{'plan.json':plan},{'outcome_blind':True,'plan_id':identity})
    pointer={'state':'FROZEN_OUTCOME_BLIND_SUCCESSOR_PLAN','plan_id':identity,
             'materialized_config_sha256':file_hash(config),
             'relative_bundle':str(bundle.relative_to(runtime)),'receipt_id':digest(receipt),
             'created_at':now(),'reserved_access':False,'qualified_for_final':False}
    path=repo/'reports/successor_execution_plan.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['plan_id']!=identity:raise PermissionError('INTEGRITY_OR_HASH_GATE: frozen successor plan changed')
        return old
    atomic_json(path,pointer);return pointer


def load_frozen_plan(repo,runtime):
    pointer=json.loads((Path(repo)/'reports/successor_execution_plan.json').read_text())
    bundle=(Path(runtime)/pointer['relative_bundle']).resolve()
    if not bundle.is_relative_to(Path(runtime).resolve()):raise PermissionError('Frozen plan outside runtime')
    receipt=validate_bundle(bundle)
    plan=json.loads((bundle/'plan.json').read_text())
    config=Path(repo)/'configs/successor_full_study_v31.json'
    if not config.exists() or file_hash(config)!=pointer['materialized_config_sha256'] or json.loads(config.read_text())!=plan:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: materialized frozen config changed')
    if digest(receipt)!=pointer['receipt_id'] or digest(plan)!=pointer['plan_id'] or plan!=execution_plan(repo):
        raise PermissionError('INTEGRITY_OR_HASH_GATE: execution plan differs')
    return plan,pointer


def validate_pilot(repo,runtime,track):
    repo=Path(repo);runtime=Path(runtime);path=repo/'reports'/('successor_gpu_pilot_'+track.replace('-','_')+'.json')
    doc=json.loads(path.read_text());protocol=load_protocol(repo)
    if doc.get('track')!=track or doc.get('state')!='SUCCEEDED_MEASURED_SUCCESSOR_PILOT':raise ValueError('Pilot not successful')
    if doc.get('families')!=FAMILIES or doc.get('fixed_settings')!=FIXED or doc.get('seed')!=11 or doc.get('year')!=2022:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: pilot recipe differs')
    if doc.get('information_set')!=('I3' if track=='Main-A' else 'I4') or doc.get('reserved_access') is not False:
        raise PermissionError('Pilot information/scope mismatch')
    if doc.get('protocol_sha256')!=file_hash(repo/'configs/completion_extension_v31.json'):raise PermissionError('Pilot protocol changed')
    if len(doc.get('results',[]))!=6 or {r['family'] for r in doc['results']}!=set(FAMILIES):raise ValueError('Incomplete pilot family bill')
    for row in doc['results']:
        if row['state']!='SUCCEEDED':raise ValueError('Incompatible pilot fit')
        bundle=runtime/'artifacts/track_development'/row['run_id'];receipt=validate_bundle(bundle)
        metric=json.loads((bundle/'metrics.json').read_text())
        if metric!=row:raise PermissionError('Pilot result differs from immutable metric')
        model=(runtime/row['model_relative_bundle']).resolve()
        if not model.is_relative_to(runtime.resolve()) or digest(validate_bundle(model))!=row['model_receipt_id']:
            raise PermissionError('Pilot model receipt mismatch')
    return doc


def measured_admission(plan,costs,remaining=21600):
    """Pure resource admission: scores are neither accepted nor read."""
    if set(costs)!={'Main-A','Nested-B'}:raise ValueError('Both measured pilot bills required')
    projections={}
    for name,design in plan['plans'].items():
        estimate=0.;parts=[]
        for track,grid in plan['tracks'].items():
            if set(costs[track])!=set(FAMILIES):raise ValueError('Complete family costs required')
            for family,seconds in costs[track].items():
                if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite measured charge required')
                count=len(grid['outer_years'])*len(grid['information_sets'])*(design.get('trials',0)+len(plan['seeds']))
                factor=16. if name=='registered_hpo' and family not in {'lightgbm','xgboost'} else 1.
                projected=seconds*count*factor*plan['cost_rule']['margin_multiplier']
                parts.append({'track':track,'family':family,'planned_fits':count,'pilot_seconds':seconds,'projected_seconds':projected})
                estimate+=projected
        projections[name]={'conservative_projected_gpu_seconds':estimate,'parts':parts,'planned_fit_count':design['planned_fit_count']}
    admitted=next((name for name in plan['plans'] if projections[name]['conservative_projected_gpu_seconds']<=remaining),None)
    return {'state':'ADMITTED_MEASURED_SUCCESSOR_BILL' if admitted else 'RESOURCE_OR_BUDGET_GATE',
            'admitted_plan':admitted,'projections':projections,'remaining_full_gpu_seconds':remaining,
            'full_gpu_seconds_ceiling':21600,'safety_margin_multiplier':plan['cost_rule']['margin_multiplier'],
            'outcome_blind':True,'reserved_access':False,'qualified_for_final':False}


def verified_pilot_retry(existing_receipt, prior_results, incident):
    if not existing_receipt and not prior_results:return 0
    if incident.get('state')!='RESOLVED_VERIFIED' or incident.get('repair_attempt') not in (1,2):
        raise PermissionError('Existing pilot attempt preserved; verified bounded repair required')
    return incident['repair_attempt']


def classify_failure(message):
    text=message.lower()
    for tokens,category in [
        (['hash','checksum','immutable','receipt mismatch'],'INTEGRITY_OR_HASH_GATE'),
        (['reserved','authorization','auth_gate','freeze'],'AUTH_GATE'),
        (['paused_budget','budget','ceiling'],'RESOURCE_OR_BUDGET_GATE'),
        (['blocked_data','maturity','pit','source consistency','fold failure'],'SCIENTIFIC_OR_DATA_GATE'),
        (['cuda lease held','unrelated cuda owner','blocked_resource'],'RESOURCE_CONTENTION'),
        (['connectionreseterror','timed out network'],'SAFE_TRANSIENT_RETRY'),
        (['cuda error','cuda out of memory','[cuda] an illegal memory access','cublas','cudnn'],'CUDA_RUNTIME_OR_LIBRARY_DEFECT'),
        (['attributeerror','nameerror','modulenotfounderror'],'CODE_ORCHESTRATION_DEFECT')]:
        if any(token in text for token in tokens):return category
    return 'UNKNOWN_UNSAFE'


def conditional_family_gate(training,gate):
    """Support counts only, never predictive scores or pooled asset sample size."""
    observed=training.loc[training.y.notna() & training.context_eligible]
    dates=int(observed.decision_time.nunique());assets=int(observed.asset.nunique())
    admitted=dates>=gate['minimum_independent_mature_decision_dates'] and assets>=gate['minimum_assets_with_target_coverage']
    return {'state':'ADMITTED_CONDITIONAL_FAMILY' if admitted else 'NOT_ADMITTED_CONDITIONAL_FAMILY',
            'independent_mature_decision_dates':dates,'covered_assets':assets,'outcome_blind':True,
            'minimum_independent_mature_decision_dates':600,'minimum_assets_with_target_coverage':4}
