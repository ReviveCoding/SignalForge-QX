"""Canonical evidence validation and release-aligned Auxiliary-C development rows."""
import numpy as np
import pandas as pd
import re
from .runtime import digest,file_hash
from .pit import asof_snapshot


def validate_events(events,evidence_registry=None):
    required={'entity','source','field','reference_time','available_at','value','unit','raw_hash','pit_tier'}
    if not required<=set(events):
        raise ValueError('Canonical fields missing')
    if not events.pit_tier.isin(['A','B','C']).all():
        raise ValueError('Unknown PIT tier')
    if not events.raw_hash.map(lambda h:isinstance(h,str) and bool(re.fullmatch('[a-f0-9]{64}',h))).all():
        raise ValueError('Canonical raw SHA256 required')
    values=pd.to_numeric(events.value,errors='raise')
    if np.isinf(values).any() or events.unit.isna().any() or events.unit.eq('').any():
        raise ValueError('Finite-or-missing values and explicit units required')
    strict=events.pit_tier.eq('A')
    if strict.any():
        for field in ['public_evidence_hash','original_vintage_hash']:
            if field not in events or events.loc[strict,field].isna().any():
                raise ValueError('Tier A lacks timing or original version evidence')
        if evidence_registry is None:raise ValueError('Tier A requires a verified public evidence registry, not hash labels alone')
        from .sources import allowed_url
        for row in events.loc[strict].to_dict('records'):
            for field in ['public_evidence_hash','original_vintage_hash']:
                evidence=evidence_registry.get(row[field],{})
                if (evidence.get('qualification_state')!='QUALIFIED_ORIGINAL_PUBLICATION' or
                    not allowed_url(evidence.get('source_url','')) or not evidence.get('path') or
                    file_hash(evidence['path'])!=row[field]):
                    raise ValueError('Tier A public/original evidence is missing, corrupt or unqualified')
                known=pd.Timestamp(evidence['public_available_at'])
                if known.tzinfo is None or known>pd.Timestamp(row['available_at']):
                    raise ValueError('Tier A clock precedes documented public availability')
    if events.groupby(['entity','source','field']).unit.nunique().max()>1:
        raise ValueError('Conflicting canonical units')
    asof_snapshot(events,'2100-01-01T00:00Z')
    return True


def valid_mapping(mappings,entity,reference,decision):
    reference,decision=pd.Timestamp(reference),pd.Timestamp(decision)
    if reference.tzinfo is None or decision.tzinfo is None:
        raise ValueError('Mapping times must be aware')
    frame=mappings[mappings.entity.eq(entity)].copy()
    for col in ['valid_from','valid_to','known_at']:
        frame[col]=pd.to_datetime(frame[col],utc=True)
    matches=frame[(frame.valid_from<=reference)&(reference<frame.valid_to)&(frame.known_at<=decision)]
    if len(matches)!=1:
        raise ValueError('Unknown or ambiguous as-of entity mapping')
    return matches.iloc[0].mapped_entity


def eia_events(records):
    events=[]
    for record in records:
        if record['state']!='SUCCEEDED':
            continue
        event=record['event']
        for name,series in event['series'].items():
            for field in ['value','change']:
                events.append({'entity':'US','source':'eia','field':name+'_'+field,
                               'reference_time':event['reference_time'],'available_at':event['available_at'],
                               'value':series[field],'unit':event['unit'],'pit_tier':event['pit_tier'],
                               'raw_hash':record['raw']['sha256'],'clock_evidence':event['clock_evidence']})
    frame=pd.DataFrame(events)
    if len(frame):
        validate_events(frame)
    return frame


def auxiliary_rows(records):
    """Friday 18:00 decisions predict next archive issue, never the already-public change."""
    qualified=[r for r in records if r['state']=='SUCCEEDED']
    qualified.sort(key=lambda r:pd.Timestamp(r['event']['available_at']))
    rows=[]
    if not qualified:return pd.DataFrame(rows)
    def issue_date(record):
        event=record.get('event',{})
        if event.get('issue_date'):return pd.Timestamp(event['issue_date']).date()
        match=re.search(r'/([0-9]{4}_[0-9]{2}_[0-9]{2})/',record.get('release_page',''))
        if match:return pd.Timestamp(match[1].replace('_','-')).date()
        if event.get('available_at'):return pd.Timestamp(event['available_at']).tz_convert('America/New_York').date()
        raise ValueError('Discovered issue date metadata absent')
    def publication_bound(record):
        if record.get('event',{}).get('available_at'):return pd.Timestamp(record['event']['available_at'])
        return (pd.Timestamp(issue_date(record))+pd.Timedelta(days=1)).tz_localize('America/New_York')
    discovered=sorted(records,key=issue_date)
    start=pd.Timestamp(qualified[0]['event']['available_at']).tz_convert('America/New_York').normalize()
    end=max(publication_bound(record) for record in discovered).tz_convert('America/New_York').normalize()
    end+=pd.Timedelta(days=(4-end.weekday())%7)
    for origin in pd.date_range(start,end,freq='W-FRI'):
        decision=origin+pd.Timedelta(hours=18)
        known=[r for r in qualified if pd.Timestamp(r['event']['available_at'])<=decision]
        future=[r for r in discovered if issue_date(r)>decision.date()]
        if not known:continue
        current=max(known,key=lambda r:(pd.Timestamp(r['event']['reference_time']),pd.Timestamp(r['event']['available_at'])))
        next_issue=future[0] if future else None
        e=current['event'];available=pd.Timestamp(e['available_at'])
        next_available=publication_bound(next_issue) if next_issue else None
        next_issue_date=issue_date(next_issue) if next_issue else None
        if not available<=decision or (next_available is not None and not decision<next_available):
            raise ValueError('Invalid weekly as-of target chronology')
        parsed=next_issue is not None and next_issue['state']=='SUCCEEDED'
        n=next_issue['event'] if parsed else None
        if parsed and (next_issue['raw']['sha256'] in {r['raw']['sha256'] for r in known} or
            pd.Timestamp(n['reference_time'])<=pd.Timestamp(e['reference_time'])):
            raise ValueError('Source integrity: future target repeats an already-known raw version/reference')
        target_eligible=parsed and (next_available-decision)<=pd.Timedelta(days=10)
        x=[]
        for name in sorted(e['series']):
            x.extend([e['series'][name]['value'],e['series'][name]['change']])
        reference=pd.Timestamp(e['reference_time'])
        x.extend([(decision.tz_convert('UTC')-available).total_seconds()/86400,
                  (decision.tz_convert('UTC')-reference).total_seconds()/86400,
                  np.sin(2*np.pi*decision.dayofyear/365.25),np.cos(2*np.pi*decision.dayofyear/365.25)])
        rows.append({'decision_time':decision.tz_convert('UTC').isoformat(),'label_start':decision.tz_convert('UTC').isoformat(),
                     'label_end':next_available.isoformat() if next_available is not None else None,
                     'label_available_at':next_available.isoformat() if next_available is not None else None,
                     'asset':'US_COMMERCIAL_CRUDE','y':n['series']['Commercial (Excluding SPR)']['change'] if target_eligible else np.nan,
                     'x':x,'raw_hash':current['raw']['sha256'],'target_raw_hash':next_issue.get('raw',{}).get('sha256') if next_issue else None,
                     'max_dependency_available_at':e['available_at'],'pit_tier':'B',
                     'target_issue_date':str(next_issue_date) if next_issue_date else None,
                     'target_status':('archived_reported_change_original_vintage_unverified' if target_eligible else
                                      'MISSING_NEXT_RELEASE_METADATA' if next_issue is None else
                                      'FAILED_SOURCE_TARGET' if not parsed else 'MISSING_WITHIN_REGISTERED_HORIZON')})
    return pd.DataFrame(rows)
