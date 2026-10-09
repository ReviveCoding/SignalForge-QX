"""Conservative remaining-work bill from exact-input committed cache proofs."""
import json
from .runtime import digest,validate_bundle


def remaining_fit_counts(runtime, families, years, seeds, grids, parents):
    counts={}; evidence={}
    for family in families:
        missing=set();cached=0
        for year in years:
            scores=[]
            for index,trial in enumerate(grids[family]):
                key=digest({'partition':'inner_validation','year':year,'family':family,'seed':11,'trial':trial})
                parent=parents.get(key)
                if parent:
                    directory=runtime/'artifacts/auxiliary'/parent
                    validate_bundle(directory)
                    metric=json.loads((directory/'metrics.json').read_text())
                    scores.append((metric['pinball'],index));cached+=1
                else:missing.add(key)
            # If any inner trial is uncompleted, all outer fits remain reserved.
            chosen=grids[family][min(scores)[1]] if len(scores)==len(grids[family]) else None
            for seed in seeds:
                key=digest({'partition':'outer_development','year':year,'family':family,'seed':seed,'trial':chosen})
                if chosen is not None and key in parents:
                    validate_bundle(runtime/'artifacts/auxiliary'/parents[key]);cached+=1
                else:missing.add(key)
        for year in list(years)[:2]:
            for seed in seeds:
                key=digest({'partition':'inner_validation','year':year,'family':family,'seed':seed,'trial':grids[family][-1]})
                if key not in parents:missing.add(key)
        counts[family]=len(missing)
        evidence[family]={'remaining_reserved_fits':len(missing),'verified_cached_inner_outer_fits':cached,
                         'selection':'completed earlier-inner scores only; incomplete inner grid reserves all outer seeds'}
    return counts,evidence
