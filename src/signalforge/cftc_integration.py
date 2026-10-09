"""Outcome-blind CFTC positioning integration for Main/Nested development.

Historical compressed archives are not original-publication bytes. The release
clock is therefore a conservative Tier-B reconstruction with explicit known
incident exceptions; it never upgrades archive history to Tier A.
"""
import datetime,json
from pathlib import Path
import pandas as pd
from .runtime import file_hash,digest,commit_bundle,atomic_json,now
from .sources import parse_cftc
from .positioning import positioning_features,positioning_events
from .data import validate_events


def load_contract(repo):
    repo=Path(repo);path=repo/'configs/cftc_asset_mapping_v31.json';contract=json.loads(path.read_text())
    if contract.get('schema')!='cftc_asset_mapping_v31' or contract.get('selection')!='outcome_blind_domain_proxy':
        raise ValueError('Registered CFTC mapping contract required')
    if contract.get('positioning_is_cash_flow') is not False or contract.get('pit_tier')!='B':
        raise PermissionError('CFTC positioning claim boundary changed')
    return contract,path


def release_evidence(report_dates,contract,evidence_hash):
    policy=contract['release_policy'];blackout=policy['blackout_report_dates'];known=policy['known_2023_ion_releases'];corrections=policy['known_correction_upper_bounds']
    start=pd.Timestamp(blackout['start']).date();end=pd.Timestamp(blackout['end']).date();out={};blocked=[]
    for raw in sorted({pd.Timestamp(d).date() for d in report_dates}):
        key=raw.isoformat()
        if start<=raw<=end:
            blocked.append({'report_date':key,'reason':'HISTORICAL_SHUTDOWN_RELEASE_DATE_NOT_AUTHENTICATED'});continue
        if key in known:available=pd.Timestamp(known[key]).tz_convert('UTC')
        elif key in corrections:available=pd.Timestamp(corrections[key]).tz_convert('UTC')
        else:
            date=raw+datetime.timedelta(days=7)
            available=pd.Timestamp(datetime.datetime.combine(date,datetime.time(23,59,59)),tz='America/New_York').tz_convert('UTC')
        if available<=pd.Timestamp(raw,tz='UTC'):raise ValueError('CFTC release reconstruction precedes report date')
        out[key]={'available_at':available.isoformat(),'clock_evidence_hash':evidence_hash,'pit_tier':'B'}
    return out,blocked


def build_canonical(repo,runtime):
    repo,runtime=Path(repo),Path(runtime);contract,contract_path=load_contract(repo);contract_hash=file_hash(contract_path)
    acquisition_path=repo/'reports/cftc_development_acquisition.json'
    if not acquisition_path.exists():raise RuntimeError('BLOCKED_DATA: CFTC development acquisition absent')
    acquisition=json.loads(acquisition_path.read_text())
    records=[r for r in acquisition.get('records',[]) if r.get('state')=='SUCCEEDED']
    if len(records)!=28:raise RuntimeError('BLOCKED_DATA: complete 2010-2023 CFTC family archive accounting required')
    by_family={'tff':[],'disaggregated':[]}
    for record in records:by_family[record['family']].append(record)
    all_events=[];blocked=[];coverage={};archive_hashes=[]
    for asset,spec in contract['assets'].items():
        family=spec['family'];code=spec['contract_code'];asset_rows=[];asset_dates=[];names=set();seen_archives=[]
        for record in sorted(by_family[family],key=lambda r:r['year']):
            raw=record['raw'];path=Path(raw['path'])
            if file_hash(path)!=raw['sha256']:raise ValueError('CFTC immutable raw checksum mismatch')
            frame=parse_cftc(path,family);selected=frame[frame.CFTC_Contract_Market_Code.eq(code)].copy()
            if selected.empty:continue
            unexpected=set(selected.Market_and_Exchange_Names)-set(spec['expected_market_names'])
            if unexpected:raise ValueError(f'CFTC mapped market-name drift for {asset}: {sorted(unexpected)}')
            names.update(selected.Market_and_Exchange_Names.unique());asset_dates.extend(selected.report_date.tolist());seen_archives.append(raw['sha256'])
            features=positioning_features(selected,family)
            clocks,release_blocked=release_evidence(features.report_date.tolist(),contract,contract_hash)
            events,event_blocked=positioning_events(features,raw['sha256'],clocks)
            keep=set(contract['feature_fields'][family]);events=events[events.field.isin(keep)].copy()
            if len(events):
                events['contract_code']=code;events['contract_family']=family;events['entity']=asset
                events['positioning_is_cash_flow']=False;events['original_publication_qualified']=False
                asset_rows.append(events)
            blocked.extend([{**x,'asset':asset,'contract_code':code} for x in release_blocked])
            blocked.extend([{**x,'asset':asset,'contract_code':code} for x in event_blocked])
        if not asset_rows:raise RuntimeError('BLOCKED_DATA: no mapped CFTC rows for '+asset)
        frame=pd.concat(asset_rows,ignore_index=True);all_events.append(frame);archive_hashes.extend(seen_archives)
        coverage[asset]={'family':family,'contract_code':code,'rows':len(frame),'report_dates':len(set(asset_dates)),
            'first_report':min(asset_dates).date().isoformat(),'last_report':max(asset_dates).date().isoformat(),
            'market_names':sorted(names),'archive_hashes':sorted(set(seen_archives))}
    events=pd.concat(all_events,ignore_index=True).sort_values(['available_at','entity','field','reference_time']).reset_index(drop=True)
    validate_events(events)
    # No contract/date/field can map twice after canonicalization.
    if events.duplicated(['entity','field','reference_time','available_at']).any():raise ValueError('Duplicate canonical CFTC asset event')
    payload=json.loads(events.to_json(orient='records'))
    identity=digest({'contract_hash':contract_hash,'archive_hashes':sorted(set(archive_hashes)),'events':payload})
    folder=runtime/'artifacts/cftc_canonical'/identity
    commit_bundle(folder,{'events.json':payload,'coverage.json':coverage,'blocked.json':blocked},{'contract_hash':contract_hash,'pit_tier':'B','qualified_for_final':False})
    report={'state':'SUCCEEDED_RECONSTRUCTED_CFTC_POSITIONING','artifact_id':identity,'contract_sha256':contract_hash,
        'events_relative_path':str((folder/'events.json').relative_to(runtime)),'events_sha256':file_hash(folder/'events.json'),
        'rows':len(events),'assets':coverage,'blocked_release_rows':len(blocked),'pit_tier':'B','positioning_is_cash_flow':False,
        'original_publication_qualified':False,'qualified_for_final':False,'reserved_access':False,'created_at':now(),
        'claim_boundary':contract['claim_boundary']}
    atomic_json(repo/'reports/cftc_canonical_integration.json',report);return report
