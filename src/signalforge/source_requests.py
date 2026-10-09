"""Registered bounded keyed requests; authentication never qualifies source science."""
import json,os,re
from pathlib import Path
import pandas as pd
from .runtime import paths,digest,atomic_json,commit_bundle,file_hash,now
from .sources import Acquisition,INDEX,parse_fred

FRED_MAX_JSON_VINTAGES=2000
FRED_SEGMENT_MAX_CALENDAR_DAYS=1750
FRED_MAX_REGISTERED_SEGMENTS=25


def fred_plan_path(repo):
    repo=Path(repo)
    successor=repo/'.local/fred_request_plan_v31.json'
    legacy=repo/'.local/fred_request_plan.json'
    return successor if successor.is_file() else legacy


def date(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):raise ValueError('Explicit calendar date required')
    return pd.Timestamp(value)


def validate_fred_plan(plan,require_json_safe_segments=False):
    if set(plan)!={'source','partition','reserved_access','registered_at','request_selection','max_bytes','max_pages_per_series','page_size','series'}:
        raise ValueError('Unknown or missing FRED plan fields')
    if plan.get('source')!='fred' or plan.get('partition')!='development' or plan.get('reserved_access') is not False:
        raise PermissionError('Explicit development-only FRED request plan required')
    registered=pd.Timestamp(plan['registered_at'])
    if registered.tzinfo is None or registered>pd.Timestamp.now(tz='UTC'):raise ValueError('Actual aware request registration time required')
    if plan.get('request_selection')!='outcome_blind':raise PermissionError('Outcome-blind source request selection required')
    if any(key in plan for key in ['api_key','apikey','token','key']):raise PermissionError('Credentials belong only in the process environment')
    for key,ceiling in [('max_bytes',16*1024**2),('max_pages_per_series',100),('page_size',1000)]:
        if type(plan.get(key)) is not int or not 0<plan[key]<=ceiling:raise ValueError('Bounded FRED request budget required: '+key)
    series=plan.get('series',[])
    if not 1<=len(series)<=FRED_MAX_REGISTERED_SEGMENTS or not 1<=len({s['series_id'] for s in series})<=5:
        raise ValueError('One to five pilot series with a bounded segmented request plan required')
    for s in series:
        if set(s)!={'series_id','unit','native_frequency','observation_start','observation_end','realtime_start','realtime_end','output_type'}:
            raise ValueError('Unknown or missing FRED series request fields')
        if not re.fullmatch(r'[A-Z0-9_.]{1,32}',s['series_id']):raise ValueError('Invalid explicit FRED series')
        if not isinstance(s.get('unit'),str) or not 0<len(s['unit'])<=200 or s.get('native_frequency') not in {'D','W','M','Q','A'}:raise ValueError('Unit/native frequency metadata required')
        if any(key in s for key in ['api_key','apikey','token','key']):raise PermissionError('Credential-bearing plan forbidden')
        start,end=date(s['observation_start']),date(s['observation_end'])
        first,last=date(s['realtime_start']),date(s['realtime_end'])
        if start>end or first>last or end>=pd.Timestamp('2024-01-01') or last>=pd.Timestamp('2024-01-01'):
            raise PermissionError('Observation and vintage request ranges must exclude reserved dates')
        if s.get('output_type') not in {1,3}:raise ValueError('Registered real-time snapshot or new/revised output required')
        if s['output_type']==1 and first!=last:raise ValueError('Initial as-of snapshot must use one explicit vintage date')
    common_revision_end=set()
    for identifier in {s['series_id'] for s in series}:
        requests=sorted([s for s in series if s['series_id']==identifier],key=lambda s:s['realtime_start'])
        if len({(s['unit'],s['native_frequency']) for s in requests})!=1:raise ValueError('Same series cannot change unit/frequency between vintages')
        if any(date(right['realtime_start'])<=date(left['realtime_end']) for left,right in zip(requests,requests[1:])):
            raise ValueError('Snapshot/revision vintage intervals must be disjoint')
        snapshots=[s for s in requests if s['output_type']==1];revisions=[s for s in requests if s['output_type']==3]
        if len(snapshots)!=1:raise ValueError('Each FRED series requires exactly one initial snapshot')
        snapshot=snapshots[0]
        if requests[0] is not snapshot:raise ValueError('Initial FRED snapshot must precede revision segments')
        if revisions:
            expected=date(snapshot['realtime_end'])+pd.Timedelta(days=1)
            for segment in revisions:
                start,end=date(segment['realtime_start']),date(segment['realtime_end'])
                if start!=expected:raise ValueError('FRED revision segments must be contiguous after the initial snapshot')
                if require_json_safe_segments and (end-start).days+1>FRED_SEGMENT_MAX_CALENDAR_DAYS:
                    raise ValueError('FRED revision segment exceeds conservative JSON vintage-date safety bound')
                expected=end+pd.Timedelta(days=1)
            common_revision_end.add(revisions[-1]['realtime_end'])
    if common_revision_end and len(common_revision_end)!=1:raise ValueError('All FRED series must share the same development vintage end')
    return digest(plan)


def decode_fred_page(raw,series,expected_offset,page_size,expected_count=None):
    if file_hash(raw['path'])!=raw['sha256']:raise ValueError('FRED raw source changed')
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate FRED JSON field')
            result[key]=value
        return result
    body=json.loads(Path(raw['path']).read_text(),object_pairs_hook=unique_object)
    if not isinstance(body.get('observations'),list):raise ValueError('FRED observation page absent')
    if (type(body.get('offset')) is not int or type(body.get('limit')) is not int or
        body.get('offset')!=expected_offset or body.get('limit')!=page_size or
        type(body.get('count')) is not int or body['count']<expected_offset):
        raise ValueError('FRED pagination metadata mismatch')
    if expected_count is not None and body['count']!=expected_count:raise ValueError('FRED count changed during snapshot acquisition')
    if int(body.get('output_type',0))!=series['output_type'] or body.get('units')!='lin':raise ValueError('FRED output/unit transformation mismatch')
    observations=body['observations']
    if len(observations)>page_size or len(observations)>body['count']-expected_offset:raise ValueError('FRED page exceeds registered count')
    window=min(page_size,body['count']-expected_offset)
    if series['output_type']==1 and len(observations)<window:raise ValueError('FRED truncated page')
    # Vintage output is a sparse observation-date x vintage-date cross-tab.
    # The provider count need not equal emitted rows (actual DGS10: 4002 vs
    # 1251). Nonempty pages advance by date rows, never cells or requested
    # limit: probe the immediate continuation after any short array. Empty
    # pages attest an empty requested window; retain that receipt and scan
    # onward through stable count, rather than treating sparsity as EOF.
    progress=len(observations) if observations else window
    labels=[date(row['date']) for row in observations]
    if any(a>=b for a,b in zip(labels,labels[1:])):raise ValueError('FRED observation rows not strictly ascending')
    output=[];start,end=date(series['observation_start']),date(series['observation_end'])
    first,last=date(series['realtime_start']),date(series['realtime_end'])
    if body.get('realtime_start')!=series['realtime_start'] or body.get('realtime_end')!=series['realtime_end']:
        raise ValueError('FRED response real-time envelope differs from explicit request')
    if series['output_type']==1:
        for row in observations:
            if row.get('realtime_end')!=series['realtime_end']:raise ValueError('Requested as-of snapshot end clock mismatch')
            value=row.get('value')
            if not isinstance(value,str):raise ValueError('Malformed FRED snapshot numerical value')
            try:
                if value!='.' and not __import__('math').isfinite(float(value)):raise ValueError()
            except (ValueError,OverflowError):raise ValueError('Malformed or nonfinite FRED snapshot numerical value') from None
        output=parse_fred(raw['path'],series['native_frequency'])
        for r in output:
            if date(r['realtime_start'])!=first:raise ValueError('Requested as-of snapshot clock mismatch')
            # Snapshot bounds do not assert the observation was first published
            # on that date. Its earlier original release remains unqualified.
            r['clock_evidence']='explicit_date_only_asof_snapshot_not_first_publication'
    else:
        prefixes=[series['series_id']+'_']
        if series['series_id'][0].isdigit():prefixes.append('_'+series['series_id']+'_')
        for row in observations:
            reference=date(row['date'])
            for key,value in row.items():
                if key=='date':continue
                matches=[p for p in prefixes if key.startswith(p)]
                if len(matches)!=1:raise ValueError('Unexpected FRED vintage column')
                prefix=matches[0]
                suffix=key[len(prefix):]
                if re.fullmatch(r'\d{8}',suffix):suffix=suffix[:4]+'-'+suffix[4:6]+'-'+suffix[6:]
                vintage=date(suffix)
                if not first<=vintage<=last:raise PermissionError('Response vintage outside registered development request')
                if not isinstance(value,str):raise ValueError('Malformed FRED vintage numerical value')
                try:number=None if value=='.' else float(value)
                except (ValueError,OverflowError):raise ValueError('Malformed FRED vintage numerical value') from None
                if number is not None and not __import__('math').isfinite(number):raise ValueError('Nonfinite FRED vintage value')
                period=reference+pd.offsets.MonthEnd(0) if series['native_frequency']=='M' else reference
                output.append({'reference_label':row['date'],'reference_time':period.tz_localize('UTC').isoformat(),
                    'realtime_start':str(vintage.date()),'realtime_end':None,'value':number,'pit_tier':'B',
                    'available_at':(vintage+pd.Timedelta(days=1,hours=12)).tz_localize('UTC').isoformat(),
                    'clock_evidence':'date_only_database_vintage_not_authenticated_source_publication','original_publication_qualified':False})
    for row in observations:
        if not start<=date(row['date'])<=end:raise PermissionError('Response reference outside registered development request')
    for r in output:
        if series['native_frequency'] in {'Q','A'}:
            period=date(r['reference_label'])
            period+=pd.offsets.QuarterEnd(startingMonth=12) if series['native_frequency']=='Q' else pd.offsets.YearEnd()
            r['reference_time']=period.tz_localize('UTC').isoformat()
        r.update(series_id=series['series_id'],unit=series['unit'],raw_hash=raw['sha256'],source='fred',model_qualified=False)
    if len({(r['reference_label'],r['realtime_start']) for r in output})!=len(output):raise ValueError('Duplicate FRED reference/vintage')
    if file_hash(raw['path'])!=raw['sha256']:raise ValueError('FRED raw changed during parsing')
    return output,body['count'],progress


def acquire_fred_plan(repo,runtime,plan,client=None):
    plan_id=validate_fred_plan(plan,require_json_safe_segments=True)
    key=os.environ.get('FRED_API_KEY')
    if not key:raise PermissionError('BLOCKED_AUTH: FRED_API_KEY absent')
    if not re.fullmatch(r'[a-z0-9]{32}',key):
        raise PermissionError('BLOCKED_AUTH: FRED_API_KEY must be 32 lowercase alphanumeric characters')
    client=client or Acquisition(runtime,plan['max_bytes']);records=[];raws=[];segments=[]
    # Commit the outcome-blind request definition before fetching any values.
    commit_bundle(runtime/'artifacts/fred_request_plans'/plan_id,{'plan.json':plan},{'reserved_access':False})
    for series in plan['series']:
        offset=0;count=None;pages=[];previous_reference=None
        for page in range(plan['max_pages_per_series']):
            if len(raws)>=100:raise RuntimeError('BLOCKED_RESOURCE: total FRED pilot page ceiling 100 reached')
            params={k:series[k] for k in ['series_id','observation_start','observation_end','realtime_start','realtime_end','output_type']}
            params.update(api_key=key,file_type='json',units='lin',sort_order='asc',offset=offset,limit=plan['page_size'])
            raw=client.cached(INDEX['fred'],'json',params) or client.fetch(INDEX['fred'],'json',params,
                {'request_plan_id':plan_id,'series_id':series['series_id'],'page':page,'reserved_access':False})
            values,count,progress=decode_fred_page(raw,series,offset,plan['page_size'],count)
            references=sorted({r['reference_label'] for r in values})
            if references and previous_reference is not None and references[0]<=previous_reference:
                raise ValueError('Overlapping or unordered FRED pagination windows')
            if references:previous_reference=references[-1]
            pages.append({'offset':offset,'covered_offset_end':offset+progress,
                          'observation_rows':len(references),'observation_versions':len(values),
                          'raw_sha256':raw['sha256']})
            records.extend(values);raws.append(raw);offset+=progress
            if offset==count:break
        else:raise RuntimeError('BLOCKED_RESOURCE: registered FRED page ceiling reached; no partial completion')
        segments.append({'series_id':series['series_id'],'output_type':series['output_type'],
                         'realtime_start':series['realtime_start'],'realtime_end':series['realtime_end'],
                         'count':count,'covered_offset_end':offset,'pages':pages})
    keys=[(r['series_id'],r['reference_label'],r['realtime_start']) for r in records]
    if len(set(keys))!=len(keys):raise ValueError('Overlapping FRED pages or vintage requests')
    if not records:raise RuntimeError('BLOCKED_DATA: no observation versions returned; empty raw download is not success')
    result={'state':'SUCCEEDED_RAW_FRED_DEVELOPMENT_VINTAGES','plan_id':plan_id,'records':records,'raw_receipts':raws,
        'pagination_segments':segments,'pagination_semantics':'continuous_date_rows_with_empty_window_receipts_v1',
        'pit_tier':'B','model_qualified':False,'reserved_access':False,'created_at':now(),
        'claim_boundary':'Explicit as-of/vintage database dates and declared units; original source publication, units evidence and canonical model input qualification remain separate'}
    identity=digest(result);commit_bundle(runtime/'artifacts/fred_request_results'/identity,{'results.json':result},{'plan_id':plan_id})
    result['artifact_id']=identity;return result


def keyed_pilot(repo,runtime,source):
    repo,runtime=Path(repo),Path(runtime)
    if source=='tiingo':
        from .tiingo import register_plan,acquire_plan,build_inputs
        result=acquire_plan(repo,runtime,register_plan(repo,runtime))
        result['inputs']=build_inputs(repo,runtime)
        return result
    if source=='tiingo_actions':
        from .tiingo_actions import register_plan,acquire_plan
        result=acquire_plan(repo,runtime,register_plan(repo,runtime))
        return {'source':'tiingo_actions','state':result['state'],'artifact_id':digest(result),
            'rows':len(result['records']),'complete_payment_dates':result['complete_payment_dates'],
            'economic_qualified':False,'reserved_access':False}
    if source=='nport':
        from .sec_pilot import acquire_nport_pilot
        return acquire_nport_pilot(repo,runtime)
    if source=='fred':
        plan_path=fred_plan_path(repo)
        if plan_path.is_file():validate_fred_plan(json.loads(plan_path.read_text(encoding='utf-8-sig')),require_json_safe_segments=True)
    if source not in {'fred','prices'}:raise ValueError('Unknown keyed source')
    key='FRED_API_KEY' if source=='fred' else 'ALPHAVANTAGE_API_KEY'
    if not os.environ.get(key):raise PermissionError('BLOCKED_AUTH: '+key+' absent')
    if source=='prices':
        # The documented weekly endpoint returns 25+ years and has no date-range
        # parameter. Never silently download its reserved period at this stage.
        return {'source':'prices','state':'BLOCKED_SOURCE_TIME_SCOPE','model_qualified':False,'reserved_access':False,
            'qualification':'PRELIMINARY_LEGACY',
            'reason':'Documented weekly API has no historical end-date filter; reserved price history is not authorized before freeze. Supply separately qualified pre-2024 P0 cards. Premium daily/action requests remain unauthorized.',
            'documentation':'https://www.alphavantage.co/documentation/'}
    path=fred_plan_path(repo)
    if not path.is_file():return {'source':'fred','state':'BLOCKED_REQUEST_PLAN','reason':'Explicit bounded FRED request plan absent; no latest-value default request','reserved_access':False}
    plan=json.loads(path.read_text(encoding='utf-8-sig'));result=acquire_fred_plan(repo,runtime,plan)
    atomic_json(repo/'reports/fred_development_acquisition.json',result)
    return {'source':'fred','state':result['state'],'plan_id':result['plan_id'],'artifact_id':result['artifact_id'],
        'observation_versions':len(result['records']),'pit_tier':'B','model_qualified':False,'reserved_access':False}
