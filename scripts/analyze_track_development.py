"""Matched-date development comparisons; training seeds never multiply sample size."""
import json
import collections
import numpy as np
import pandas as pd
from signalforge.runtime import paths,validate_bundle,atomic_json,file_hash,now
from signalforge.statistics import calendar_indices,paired_statistics,exploratory_fdr
from signalforge.evaluation import holm

repo,runtime=paths();qualification=json.loads((repo/'reports/track_input_qualification_v311.json').read_text())
rows=[r for track in qualification['tracks'] for r in track['cpu_development']['results']]
full=repo/'reports/successor_gpu_full.json'
if full.exists():rows.extend(r for track in json.loads(full.read_text())['tracks'] for r in track['results'])
groups=collections.defaultdict(list)
for row in rows:
    bundle=runtime/'artifacts/track_development'/row['run_id'];validate_bundle(bundle)
    metric=json.loads((bundle/'metrics.json').read_text())
    if metric['year']>=2024:raise PermissionError('Reserved metrics forbidden')
    groups[(metric['track'],metric['information'],metric['family'])].append(metric)
tables=[];series={};seed_year=[]
for key,metrics in sorted(groups.items()):
    by_date=collections.defaultdict(list);normalizers={};seeds_by_year=collections.defaultdict(set)
    for metric in metrics:
        seeds_by_year[metric['year']].add(metric['seed'])
        prior=normalizers.setdefault(metric['year'],metric['normalizer_id'])
        if prior!=metric['normalizer_id']:raise ValueError('Seed normalizer mismatch')
        seed_year.append({**dict(zip(['track','information','family'],key)),'year':metric['year'],'seed':metric['seed'],
                          'pinball':metric['pinball'],'normalized_mse':metric['normalized_mse'],'run_id':metric['run_id'],'state':metric['state']})
        for date in metric['date_losses']:
            if date.get('pinball') is not None:by_date[date['decision_time']].append(date['pinball'])
    if any(seeds!={11,37,71} for seeds in seeds_by_year.values()):raise ValueError('Complete registered seed family required')
    if any(len(v)!=3 for v in by_date.values()):raise ValueError('Matched seed/date coverage required')
    losses={date:float(np.mean(v)) for date,v in sorted(by_date.items())};series[key]=(losses,normalizers)
    tables.append({**dict(zip(['track','information','family'],key)),'n_independent_dates':len(losses),
                   'mean_seed_averaged_date_pinball':float(np.mean(list(losses.values()))),'successful_outer_results':len(metrics),
                   'aggregation':'average seed loss within each market date, then average dates; not ensemble score',
                   'fallback_rows':sum(m.get('context_fallback_rows',0) for m in metrics)})
contrasts=[]
def contrast(left,right,identifier,kind):
    if left not in series or right not in series:return None
    a,an=series[left];b,bn=series[right]
    if list(a)!=list(b) or an!=bn:raise ValueError('Information/architecture comparison grid or normalizer mismatch')
    blocks={}
    for block in [4,8,13]:
        idx,metadata=calendar_indices(list(a),block=block,draws=2000)
        blocks[str(block)]=paired_statistics(list(a.values()),list(b.values()),idx,metadata)
    doc={'id':identifier,'kind':kind,'baseline':left,'candidate':right,'blocks':blocks,'n_independent_dates':len(a),
         'claim_boundary':'reconstructed Tier-B P0 development only; no confirmatory final, economics or causal claim'}
    contrasts.append(doc);return doc
for track in ['Main-A','Nested-B']:
    infos=['I0','I1','I2','I3']+(['I4'] if track=='Nested-B' else [])
    families=sorted({k[2] for k in series if k[0]==track})
    for family in families:
        for a,b in zip(infos,infos[1:]):contrast((track,a,family),(track,b,family),track+'_'+family+'_'+b+'_vs_'+a,'information_increment')
    reference='lightgbm' if (track,infos[-1],'lightgbm') in series else 'ridge'
    for family in families:
        if family!=reference:contrast((track,infos[-1],reference),(track,infos[-1],family),track+'_'+reference+'_minus_'+family,'architecture_development')
pvalues={c['id']:c['blocks']['8']['two_sided_centered_block_p'] for c in contrasts}
registered={'H1a':'Main-A_lightgbm_I2_vs_I1','H1b':'Main-A_lightgbm_I3_vs_I2','H1c':'Nested-B_lightgbm_I4_vs_I3'}
holm_values={h:pvalues.get(identifier) for h,identifier in registered.items()};holm_values.update(H2=None,H3=None)
report={'state':'COMPLETED_RECONSTRUCTED_DEVELOPMENT_ANALYSIS','created_at':now(),'tables':tables,'contrasts':contrasts,
        'year_seed_robustness':seed_year,'registered_Holm_development_diagnostic':holm(holm_values),
        'missing_registered_contrasts':[h for h,p in holm_values.items() if p is None],
        'exploratory_multiplicity':exploratory_fdr(pvalues),'training_seeds_are_independent_market_samples':False,
        'qualification_sha256':file_hash(repo/'reports/track_input_qualification_v311.json'),
        'gpu_full_receipt_sha256':file_hash(full) if full.exists() else None,
        'reserved_access':False,'qualified_for_final':False,'economic_qualified':False,
        'complexity_interpretation':'Architecture justification requires matched development effects, cost and robustness; pilot accuracy is never used for admission.',
        'not_executed_here':['retrained age/coverage/gate ablations','frozen dependency-closed outage interventions','selection-frozen H2 candidate']}
atomic_json(repo/'reports/track_development_analysis.json',report)
print(json.dumps({'state':report['state'],'tables':len(tables),'paired_contrasts':len(contrasts),'missing_registered_contrasts':report['missing_registered_contrasts']}))
