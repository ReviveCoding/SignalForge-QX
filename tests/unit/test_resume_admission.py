import json
from signalforge.resume_admission import remaining_fit_counts
from signalforge.runtime import digest,commit_bundle


def test_remaining_bill_reuses_exact_inner_and_selected_outer_only(tmp_path):
    parents={};grids={'ridge':[{'alpha':1},{'alpha':2}]}
    def put(partition,seed,trial,score):
        key=digest({'partition':partition,'year':2018,'family':'ridge','seed':seed,'trial':trial})
        commit_bundle(tmp_path/'artifacts/auxiliary'/key,{'metrics.json':{'pinball':score}},{})
        parents[key]=key
    put('inner_validation',11,grids['ridge'][0],2)
    put('inner_validation',11,grids['ridge'][1],1)
    put('inner_validation',37,grids['ridge'][1],1)
    put('outer_development',11,grids['ridge'][1],1)
    counts,evidence=remaining_fit_counts(tmp_path,['ridge'],[2018],[11,37],grids,parents)
    assert counts=={'ridge':1} and evidence['ridge']['verified_cached_inner_outer_fits']==3
    parents.pop(digest({'partition':'inner_validation','year':2018,'family':'ridge','seed':11,'trial':grids['ridge'][0]}))
    counts,_=remaining_fit_counts(tmp_path,['ridge'],[2018],[11,37],grids,parents)
    assert counts['ridge']==3  # missing inner plus both conservatively reserved outer fits
