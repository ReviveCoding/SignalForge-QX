"""Register deterministic FRED JSON-safe vintage segments without reading numerical values."""
import json
import pandas as pd
from signalforge.runtime import paths,atomic_json,file_hash,digest,now
from signalforge.source_requests import validate_fred_plan,FRED_SEGMENT_MAX_CALENDAR_DAYS

repo,runtime=paths()
legacy=repo/'.local/fred_request_plan.json'
successor=repo/'.local/fred_request_plan_v31.json'
if not legacy.exists():raise FileNotFoundError('Legacy outcome-blind FRED plan absent')
parent=json.loads(legacy.read_text(encoding='utf-8-sig'))

def generate(parent,registered_at):
    by={}
    for row in parent['series']:by.setdefault(row['series_id'],[]).append(row)
    segments=[]
    for series_id in sorted(by):
        rows=sorted(by[series_id],key=lambda r:r['realtime_start'])
        snapshots=[r for r in rows if r['output_type']==1];revisions=[r for r in rows if r['output_type']==3]
        if len(snapshots)!=1 or len(revisions)!=1:
            raise ValueError('Legacy FRED plan must have exactly one snapshot and one broad revision interval per series')
        snapshot=json.loads(json.dumps(snapshots[0]));broad=revisions[0];segments.append(snapshot)
        start=pd.Timestamp(broad['realtime_start']);end=pd.Timestamp(broad['realtime_end'])
        while start<=end:
            stop=min(start+pd.Timedelta(days=FRED_SEGMENT_MAX_CALENDAR_DAYS-1),end)
            segment=json.loads(json.dumps(broad));segment['realtime_start']=str(start.date());segment['realtime_end']=str(stop.date())
            segments.append(segment);start=stop+pd.Timedelta(days=1)
    plan={k:json.loads(json.dumps(v)) for k,v in parent.items() if k!='series'}
    plan['registered_at']=registered_at;plan['series']=segments
    return plan

if successor.exists():
    plan=json.loads(successor.read_text(encoding='utf-8-sig'))
    expected=generate(parent,plan['registered_at'])
    if plan!=expected:raise PermissionError('Existing segmented FRED plan differs from deterministic child of frozen parent')
    validate_fred_plan(plan,require_json_safe_segments=True)
else:
    plan=generate(parent,now());validate_fred_plan(plan,require_json_safe_segments=True);atomic_json(successor,plan)

receipt={
    'state':'PREREGISTERED_SEGMENTED_FRED_PLAN','created_at':now(),'path':str(successor),
    'plan_sha256':file_hash(successor),'plan_id':digest(plan),'parent_plan_sha256':file_hash(legacy),
    'segment_count':len(plan['series']),'series_count':len({x['series_id'] for x in plan['series']}),
    'max_revision_calendar_days':FRED_SEGMENT_MAX_CALENDAR_DAYS,'provider_json_vintage_limit':2000,
    'selection':'outcome_blind_transport_segmentation_only','reserved_access':False,
    'documentation':'https://fred.stlouisfed.org/docs/api/fred/series_observations.html'
}
atomic_json(repo/'reports/fred_segmented_plan.json',receipt);print(json.dumps(receipt,indent=2))
