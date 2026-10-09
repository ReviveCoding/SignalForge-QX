"""Development-only canonical macro integration from segmented FRED/ALFRED vintages."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import atomic_json,commit_bundle,digest,file_hash,now,validate_bundle
from .data import validate_events

LEVEL_UNITS={
    'CPIAUCSL':'Index 1982-1984=100',
    'UNRATE':'Percent',
    'DGS2':'Percent',
    'DGS10':'Percent',
}
MONTHLY={'CPIAUCSL','INDPRO','UNRATE'}
EXPECTED={'CPIAUCSL','INDPRO','UNRATE','DGS2','DGS10'}


def _stable_level_event(row):
    series=row['series_id']
    if series not in LEVEL_UNITS:raise ValueError('Series requires an explicit canonical transform: '+series)
    if row.get('unit')!=LEVEL_UNITS[series]:
        raise ValueError('Stable-level macro unit changed for '+series+'; register a unit-stable transform instead of relabeling')
    return {
        'entity':series,'source':'macro','field':'value','value':row.get('value'),'unit':LEVEL_UNITS[series],
        'reference_time':row['reference_time'],'available_at':row['available_at'],'raw_hash':row['raw_hash'],
        'pit_tier':'B','fred_series_id':series,'fred_vintage_date':row['realtime_start'],
        'clock_evidence':row.get('clock_evidence','date_only_database_vintage_not_authenticated_source_publication'),
        'original_publication_qualified':False,'unit_semantics':'registered_stable_level_development_semantics'
    }


def _indpro_yoy(records):
    """Reconstruct same-vintage YoY revisions so multiplicative index rebasing cancels."""
    state={};events={}
    frame=pd.DataFrame(records).copy()
    if frame.empty:return []
    frame['vintage']=pd.to_datetime(frame.realtime_start)
    frame['reference']=pd.to_datetime(frame.reference_time,utc=True)
    frame=frame.sort_values(['vintage','reference','raw_hash'])
    for vintage,group in frame.groupby('vintage',sort=True):
        changed=set()
        available=max(pd.Timestamp(v) for v in group.available_at)
        for row in group.to_dict('records'):
            period=pd.Timestamp(row['reference_time']).tz_convert('UTC').tz_localize(None).to_period('M')
            value=row.get('value')
            value=np.nan if value is None else float(value)
            state[period]={'value':value,'raw_hash':row['raw_hash'],'reference_time':row['reference_time']}
            changed.add(period)
        affected=set(changed)
        for period in changed:
            affected.add(period+12)
        for period in sorted(affected):
            if period not in state or period-12 not in state:continue
            numerator=state[period];denominator=state[period-12]
            a,b=numerator['value'],denominator['value']
            value=np.nan
            if np.isfinite(a) and np.isfinite(b) and b!=0:
                value=100.0*(a/b-1.0)
            dependency_hashes=sorted({numerator['raw_hash'],denominator['raw_hash']})
            lineage_hash=digest({'transform':'same_vintage_yoy_percent_v1','series':'INDPRO',
                'reference_month':str(period),'vintage':str(vintage.date()),'raw_hashes':dependency_hashes})
            event={
                'entity':'INDPRO','source':'macro','field':'yoy_percent','value':value,'unit':'percent_yoy',
                'reference_time':numerator['reference_time'],'available_at':available.isoformat(),
                'raw_hash':lineage_hash,'pit_tier':'B','fred_series_id':'INDPRO',
                'fred_vintage_date':str(vintage.date()),'dependency_raw_hashes':dependency_hashes,
                'clock_evidence':'date_only_database_vintage_not_authenticated_source_publication',
                'original_publication_qualified':False,
                'unit_semantics':'same_vintage_ratio_invariant_to_multiplicative_index_rebasing'
            }
            events[(event['reference_time'],event['available_at'])]=event
    return list(events.values())


def build_macro_integration(repo,runtime):
    repo,runtime=Path(repo),Path(runtime)
    source=repo/'reports/fred_development_acquisition.json'
    if not source.exists():raise RuntimeError('BLOCKED_DATA: successful segmented FRED acquisition receipt absent')
    acquisition=json.loads(source.read_text(encoding='utf-8-sig'))
    if acquisition.get('state')!='SUCCEEDED_RAW_FRED_DEVELOPMENT_VINTAGES':
        raise RuntimeError('BLOCKED_DATA: segmented FRED acquisition not successful')
    identity=acquisition.get('artifact_id')
    if not isinstance(identity,str) or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('FRED acquisition immutable identity absent')
    bundle=runtime/'artifacts/fred_request_results'/identity
    validate_bundle(bundle)
    immutable=json.loads((bundle/'results.json').read_text())
    if immutable!={k:v for k,v in acquisition.items() if k!='artifact_id'} or digest(immutable)!=identity:
        raise ValueError('FRED acquisition report differs from immutable source result')
    for raw in acquisition.get('raw_receipts',[]):
        if not Path(raw['path']).resolve().is_relative_to((runtime/'raw').resolve()) or file_hash(raw['path'])!=raw['sha256']:
            raise ValueError('FRED immutable raw receipt mismatch before canonicalization')
    rows=acquisition.get('records',[])
    if not rows:raise RuntimeError('BLOCKED_DATA: FRED acquisition has no observation versions')
    frame=pd.DataFrame(rows)
    if set(frame.series_id.unique())!=EXPECTED:raise ValueError('FRED canonical series set changed')
    if any(pd.Timestamp(v)>=pd.Timestamp('2024-01-01') for v in frame.realtime_start):
        raise PermissionError('Reserved FRED vintage encountered')
    if not frame.pit_tier.eq('B').all():raise ValueError('Development macro integration requires Tier-B source vintages')

    events=[]
    for series in sorted(EXPECTED-{'INDPRO'}):
        events.extend(_stable_level_event(row) for row in frame[frame.series_id.eq(series)].to_dict('records'))
    events.extend(_indpro_yoy(frame[frame.series_id.eq('INDPRO')].to_dict('records')))
    canonical=pd.DataFrame(events)
    if canonical.empty:raise RuntimeError('BLOCKED_DATA: no canonical macro events')
    keys=['entity','source','field','reference_time','available_at']
    if canonical.duplicated(keys).any():raise ValueError('Duplicate canonical macro event/version')
    validate_events(canonical)
    if set(canonical.entity.unique())!=EXPECTED:raise ValueError('Canonical macro entity coverage incomplete')

    payload=json.loads(canonical.sort_values(['available_at','entity','field','reference_time']).to_json(orient='records'))
    feature_contract=repo/'configs/track_feature_specs_v311.json'
    identity=digest({'fred_artifact_id':acquisition.get('artifact_id'),'fred_plan_id':acquisition.get('plan_id'),
        'feature_contract_sha256':file_hash(feature_contract),'events':payload})
    folder=runtime/'artifacts/macro_canonical_v311'/identity
    commit_bundle(folder,{'events.json':payload},{
        'protocol':'sgqx-v3.1.1-track-sources','pit_tier':'B','reserved_access':False,
        'original_publication_qualified':False,'feature_contract_sha256':file_hash(feature_contract)
    })
    counts={entity:int((canonical.entity==entity).sum()) for entity in sorted(EXPECTED)}
    result={
        'state':'SUCCEEDED_RECONSTRUCTED_MACRO_VINTAGES','artifact_id':identity,
        'events_relative_path':str((folder/'events.json').relative_to(runtime)),
        'events_sha256':file_hash(folder/'events.json'),'event_rows':len(canonical),'entity_counts':counts,
        'pit_tier':'B','original_publication_qualified':False,'qualified_for_final':False,
        'reserved_access':False,'feature_contract':'configs/track_feature_specs_v311.json',
        'feature_contract_sha256':file_hash(feature_contract),'created_at':now(),
        'claim_boundary':'Date-only FRED/ALFRED database vintages; INDPRO uses same-vintage YoY to avoid historical index-base mixing. No original-publication or strict-PIT claim.'
    }
    atomic_json(repo/'reports/macro_canonical_integration.json',result)
    return result
