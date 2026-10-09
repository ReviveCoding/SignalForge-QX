"""Complete synthetic grid only. Counts never become actual future evidence."""
from datetime import datetime
from data_recovery_v36.shadow import validate_message
from qualification_v35.core import clock
def completeness(rows,plan):
    seen=set();dates={t:{0:set(),1:set()} for t in ['Main-A','Nested-B']};grids={}
    for r in rows:
        validate_message(r,plan);d=clock(r['decision_at']);identity=(r['track'],d,r['asset'],r['seed'])
        if identity in seen:raise ValueError('Duplicate semantic identity')
        seen.add(identity);dates[r['track']][r['block']].add(d);grids.setdefault((r['track'],r['block'],d),set()).add((r['asset'],r['seed']))
    expected={(a,s) for a in plan['assets'] for s in plan['seeds']}
    for t,blocks in dates.items():
        if any(len(ds)!=52 for ds in blocks.values()):raise ValueError('Exactly two sets of 52 distinct dates per track')
        if max(blocks[0])>=min(blocks[1]):raise ValueError('Chronological nonoverlap')
        for b,ds in blocks.items():
            for d in ds:
                if grids[(t,b,d)]!=expected:raise ValueError('Complete asset seed grid')
    return {'state':'TEST_ONLY_COMPLETE','rows':len(rows),'per_track_dates':{t:[len(ds) for ds in blocks.values()] for t,blocks in dates.items()},'future_evidence_dates':0,'real_forecasts':0}
