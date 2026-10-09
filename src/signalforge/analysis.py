"""Date-level diagnostics from validated forecast bundles, without seed selection."""
import json
import numpy as np
from .runtime import atomic_json,validate_bundle,now,digest
from .ensemble import FrozenEnsemble
from .metrics import quantile_loss
from .models import QUANTILES
from .evaluation import compare,holm
from .statistics import calendar_indices,paired_statistics,exploratory_fdr


def analyze_auxiliary(repo,runtime):
    path=repo/'reports/auxiliary_development_results.json'
    if not path.exists():raise RuntimeError('BLOCKED_DATA: no development forecast artifacts')
    document=json.loads(path.read_text());config=document['protocol'];results=document['results']
    families=config['families'];seeds=config['seeds'];scores={f:{} for f in families};forecasts={f:[] for f in families}
    reference=config.get('diagnostic_reference','lightgbm')
    coverage={f:{'fallback_seed_dates':0,'total_seed_dates':0} for f in families};unscored=[]
    success={(r.get('outer_year'),r.get('family'),r.get('seed')):r for r in results if r['state']=='SUCCEEDED' and r.get('run_id')}
    def load(row):
        directory=runtime/'artifacts/auxiliary'/row['run_id'];receipt=validate_bundle(directory)
        return json.loads((directory/'predictions.json').read_text()),row['scale'],row['normalizer_id']
    for year in config['outer_years']:
        base=success.get((year,'historical',seeds[0]))
        if base is None:
            unscored.append({'outer_year':year,'reason':'Registered historical fallback missing; no synthetic replacement'});continue
        fallback,scale,normalizer=load(base);dates=fallback['decision_time'];y=np.array(fallback['y'])
        for family in families:
            members={};native=0
            for seed in seeds:
                row=success.get((year,family,seed))
                if row:
                    pred,other_scale,other_normalizer=load(row)
                    if pred['decision_time']!=dates or pred['y']!=fallback['y'] or other_normalizer!=normalizer or other_scale!=scale:
                        raise ValueError('Mismatched forecast grid, outcomes or shared normalizer')
                    native+=len(dates)
                else:pred=fallback
                members[str(seed)]=(np.asarray(pred['mean']),np.asarray(pred['quantiles']))
            ensemble=FrozenEnsemble([str(s) for s in seeds]);mean,q=ensemble.predict(members)
            losses=quantile_loss(y,q,QUANTILES,scale)
            coverage[family]['fallback_seed_dates']+=len(dates)*len(seeds)-native
            coverage[family]['total_seed_dates']+=len(dates)*len(seeds)
            for i,date in enumerate(dates):
                if date in scores[family]:raise ValueError('Duplicated outer decision date')
                scores[family][date]=float(losses[i])
                forecasts[family].append({'decision_time':date,'y':float(y[i]),'mean':float(mean[i]),'quantiles':q[i].tolist(),
                                         'scale':scale,'outer_year':year,'ensemble_id':ensemble.id,'normalizer_id':normalizer})
    table=[];contrasts=[]
    shared_dates=sorted(scores[reference])
    shared_blocks={block:calendar_indices(shared_dates,block=block,draws=2000) for block in [4,8,13]} if shared_dates else {}
    for family in families:
        rows=forecasts[family]
        if not rows:
            table.append({'family':family,'state':'BLOCKED_DATA'});continue
        y=np.array([r['y'] for r in rows]);mu=np.array([r['mean'] for r in rows]);q=np.array([r['quantiles'] for r in rows]);s=np.array([r['scale'] for r in rows])
        calibration={str(alpha):{'coverage':float(np.mean(y<=q[:,j])),'nominal':float(alpha)} for j,alpha in enumerate(QUANTILES)}
        table.append({'family':family,'state':'SUCCEEDED_DIAGNOSTIC','n_unique_dates':len(scores[family]),
                      'pinball':float(np.mean(list(scores[family].values()))),'normalized_mse':float(np.mean(((y-mu)/s)**2)),
                      'interval_90_coverage':float(np.mean((y>=q[:,0])&(y<=q[:,-1]))),
                      'quantile_coverage':calibration,**coverage[family],
                      'native_seed_date_coverage':1-coverage[family]['fallback_seed_dates']/coverage[family]['total_seed_dates']})
        if family!=reference:
            sensitivity={}
            for block in [4,8,13]:
                try:
                    if sorted(scores[family])!=shared_dates:raise ValueError('Mismatched paired date grid')
                    indices,metadata=shared_blocks[block]
                    sensitivity[str(block)]=paired_statistics([scores[reference][d] for d in shared_dates],
                                                             [scores[family][d] for d in shared_dates],indices,metadata)
                except ValueError as exc:sensitivity[str(block)]={'state':'BLOCKED_CALENDAR_GRID','reason':str(exc)}
            contrasts.append({'contrast':reference+'_minus_'+family,'scope':'exploratory_reconstructed_Auxiliary_only',
                              'positive_effect':'candidate lower loss','blocks':sensitivity,'n_market_dates':len(scores[family]),
                              'seeds_are_market_replicates':False})
    pvalues={reference+'_minus_'+family:None for family in families if family!=reference}
    for contrast in contrasts:pvalues[contrast['contrast']]=contrast['blocks']['8'].get('two_sided_centered_block_p')
    report={'created_at':now(),'state':'SUCCEEDED_DIAGNOSTIC' if table and all(r['state']=='SUCCEEDED_DIAGNOSTIC' for r in table) and not unscored else 'PARTIAL',
            'protocol_hash':digest(config),'comparison_table':table,'paired_contrasts':contrasts,'unscored_folds':unscored,
            'confirmatory_Holm':holm({h:None for h in ['H1a','H1b','H1c','H2','H3']}),
            'exploratory_multiplicity':exploratory_fdr(pvalues),
            'searched_inner_recipes':sum(len(grid) for grid in config.get('trial_grids',{}).values())*len(config['outer_years']),
            'claim_boundary':'Tier B physical-only development diagnostic; no H1/H2/H3, final, economics or causal claim',
            'training_seeds':seeds,'registered_forecast_fallback':'training historical quantiles; failed seed replaced, never dropped',
            'forecast_rows':forecasts,'final_access':False}
    atomic_json(repo/'reports/auxiliary_development_analysis.json',report)
    return report
