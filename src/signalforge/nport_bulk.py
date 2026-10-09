"""Bounded 2019Q4-2023Q4 Form N-PORT bulk integration for applicable target funds only."""
import csv,datetime,json,math,os,re,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import pandas as pd
from .runtime import atomic_json,commit_bundle,digest,file_hash,now
from .sources import Acquisition,INDEX,discover_zip,validate_sec_user_agent,nport_reference_months
from .data import validate_events

TABLES=('SUBMISSION','REGISTRANT','FUND_REPORTED_INFO')


def _norm(value):
    if value is None or (isinstance(value,float) and np.isnan(value)):return ''
    return ' '.join(str(value).strip().split()).upper()


def load_plan(repo):
    repo=Path(repo);path=repo/'.local/nport_bulk_plan.json';plan=json.loads(path.read_text(encoding='utf-8-sig'))
    if plan.get('source')!='sec_nport_bulk' or plan.get('partition')!='development' or plan.get('reserved_access') is not False:
        raise PermissionError('Registered development-only N-PORT bulk plan required')
    expected=['2019Q4']+[f'{year}Q{quarter}' for year in range(2020,2024) for quarter in range(1,5)]
    if plan.get('quarters')!=expected:raise PermissionError('Outcome-blind N-PORT quarter list changed')
    if set(plan.get('targets',{}))!={'SPY','QQQ','IEF','TLT'} or sorted(plan.get('non_applicable_assets',[]))!=['GLD','SLV','UNG','USO']:
        raise PermissionError('Registered N-PORT applicability changed')
    if not 0<plan['max_bytes_per_quarter']<=768*1024**2 or not 0<plan['max_bytes_total']<=8*1024**3:
        raise ValueError('N-PORT byte budget outside preregistered ceiling')
    if not 0<plan['max_accession_clock_requests']<=256 or plan['request_rate_per_second']!=1 or plan['availability_lag_minutes']!=15:
        raise ValueError('N-PORT request/clock ceiling changed')
    t=pd.Timestamp(plan['registered_at']);
    if t.tzinfo is None:raise ValueError('Aware N-PORT plan registration required')
    return plan,path


def _read_table(archive,table):
    hits=[m for m in archive.namelist() if PurePosixPath(m).name.upper() in {table+'.TSV',table+'.TXT'}]
    if len(hits)!=1:raise ValueError('Missing or ambiguous N-PORT table: '+table)
    with archive.open(hits[0]) as stream:return pd.read_csv(stream,sep='\t',dtype=str,encoding='utf-8',low_memory=False)


def target_accessions(path,plan,quarter):
    """Discover target series from filed identity, never from later prediction outcomes."""
    with zipfile.ZipFile(path) as archive:
        submission=_read_table(archive,'SUBMISSION');registrant=_read_table(archive,'REGISTRANT');fund=_read_table(archive,'FUND_REPORTED_INFO')
    for frame,name in [(submission,'SUBMISSION'),(registrant,'REGISTRANT'),(fund,'FUND_REPORTED_INFO')]:
        if 'ACCESSION_NUMBER' not in frame or frame.ACCESSION_NUMBER.isna().any() or frame.ACCESSION_NUMBER.duplicated().any():
            raise ValueError('Missing/duplicate accession in '+name)
    required_submission={'ACCESSION_NUMBER','REPORT_DATE','FILING_DATE','SUB_TYPE'}
    required_reg={'ACCESSION_NUMBER','CIK','REGISTRANT_NAME'}
    required_fund={'ACCESSION_NUMBER','SERIES_NAME','SERIES_ID','NET_ASSETS'}|{f'{field}_MON{i}' for field in ['SALES_FLOW','REDEMPTION_FLOW','REINVESTMENT_FLOW'] for i in range(1,4)}
    if not required_submission<=set(submission) or not required_reg<=set(registrant) or not required_fund<=set(fund):raise ValueError('N-PORT bulk schema changed')
    joined=fund.merge(submission,on='ACCESSION_NUMBER',validate='one_to_one').merge(registrant,on='ACCESSION_NUMBER',validate='one_to_one')
    joined['CIK']=joined.CIK.fillna('').str.strip().str.zfill(10);joined['SERIES_NAME_NORM']=joined.SERIES_NAME.map(_norm);joined['REGISTRANT_NAME_NORM']=joined.REGISTRANT_NAME.map(_norm)
    selected=[]
    for asset,spec in plan['targets'].items():
        candidates=joined[joined.CIK.eq(spec['cik'])].copy()
        allowed_names={_norm(v) for v in spec['series_names']}
        allowed_reg={_norm(v) for v in spec['registrant_names']}
        candidates=candidates[candidates.REGISTRANT_NAME_NORM.isin(allowed_reg)]
        if spec.get('series_id'):
            candidates=candidates[candidates.SERIES_ID.fillna('').str.strip().eq(spec['series_id'])]
            candidates=candidates[candidates.SERIES_NAME_NORM.isin(allowed_names)]
        elif asset=='SPY':
            # SPY is a UIT direct registrant and currently reports no EDGAR series ID.
            candidates=candidates[candidates.SERIES_NAME_NORM.isin(allowed_names)]
        else:
            candidates=candidates[candidates.SERIES_NAME_NORM.isin(allowed_names)]
        for record in candidates.to_dict('records'):
            if record['SUB_TYPE'] not in {'NPORT-P','NPORT-P/A'}:continue
            report=pd.Timestamp(record['REPORT_DATE'])
            if pd.isna(report) or report>=pd.Timestamp('2024-01-01'):raise PermissionError('Reserved N-PORT report encountered')
            selected.append({'asset':asset,'quarter':quarter,'ACCESSION_NUMBER':record['ACCESSION_NUMBER'],'CIK':record['CIK'],
                'REGISTRANT_NAME':record['REGISTRANT_NAME'],'SERIES_NAME':record.get('SERIES_NAME'),'SERIES_ID':record.get('SERIES_ID'),
                'REPORT_DATE':record['REPORT_DATE'],'FILING_DATE':record['FILING_DATE'],'SUB_TYPE':record['SUB_TYPE'],
                'NET_ASSETS':record['NET_ASSETS'],**{f'{field}_MON{i}':record[f'{field}_MON{i}'] for field in ['SALES_FLOW','REDEMPTION_FLOW','REINVESTMENT_FLOW'] for i in range(1,4)}})
    frame=pd.DataFrame(selected)
    if len(frame) and frame.ACCESSION_NUMBER.duplicated().any():raise ValueError('One N-PORT accession mapped to multiple target assets')
    return frame


def accession_index_url(cik,accession):
    if not re.fullmatch(r'\d{10}',cik) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',accession):raise ValueError('Canonical SEC CIK/accession required')
    return f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-","")}/{accession}-index.htm'


def parse_accepted(html):
    text=' '.join(re.sub(r'<[^>]+>',' ',html).split())
    match=re.search(r'Accepted\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})',text,re.I)
    if not match:raise ValueError('SEC Accepted timestamp absent from filing index')
    local=pd.Timestamp(match.group(1)+' '+match.group(2),tz='America/New_York')
    return local.tz_convert('UTC')


def _number(value,name):
    if value is None or (isinstance(value,float) and np.isnan(value)) or str(value).strip() in {'','nan','NaN'}:return np.nan
    try:result=float(str(value).replace(',',''))
    except ValueError as error:raise ValueError('Malformed N-PORT numeric '+name) from error
    if not math.isfinite(result) or result<0:raise ValueError('Invalid N-PORT nonnegative numeric '+name)
    return result


def target_holding_summaries(path, selected):
    """Stream the holdings table and retain only target accessions; EOF read verifies that member CRC."""
    wanted={row.ACCESSION_NUMBER:row.asset for row in selected[['ACCESSION_NUMBER','asset']].itertuples(index=False)}
    if not wanted:return pd.DataFrame(columns=['ACCESSION_NUMBER','asset','holding_count','top10_abs_percentage','gross_abs_percentage'])
    values={accession:[] for accession in wanted};counts={accession:0 for accession in wanted}
    with zipfile.ZipFile(path) as archive:
        hits=[m for m in archive.namelist() if PurePosixPath(m).name.upper() in {'FUND_REPORTED_HOLDING.TSV','FUND_REPORTED_HOLDING.TXT'}]
        if len(hits)!=1:raise ValueError('Missing or ambiguous N-PORT holdings table')
        with archive.open(hits[0]) as raw:
            stream=(line.decode('utf-8') for line in raw)
            reader=csv.DictReader(stream,delimiter='\t')
            required={'ACCESSION_NUMBER','HOLDING_ID','PERCENTAGE'}
            if not required<=set(reader.fieldnames or []):raise ValueError('N-PORT holdings schema changed')
            seen_ids=set()
            for row in reader:
                accession=row.get('ACCESSION_NUMBER')
                if accession not in wanted:continue
                holding=(accession,row.get('HOLDING_ID'))
                if holding in seen_ids:raise ValueError('Duplicate target holding id')
                seen_ids.add(holding);counts[accession]+=1
                value=row.get('PERCENTAGE')
                if value in (None,''):continue
                try:pct=float(str(value).replace(',',''))
                except ValueError as error:raise ValueError('Malformed N-PORT holding percentage') from error
                if not math.isfinite(pct):raise ValueError('Non-finite N-PORT holding percentage')
                values[accession].append(abs(pct))
    result=[]
    for accession,asset in wanted.items():
        percentages=sorted(values[accession],reverse=True)
        result.append({'ACCESSION_NUMBER':accession,'asset':asset,'holding_count':counts[accession],
            'top10_abs_percentage':float(sum(percentages[:10])),'gross_abs_percentage':float(sum(percentages))})
    return pd.DataFrame(result)


def canonical_rows(selected,clocks,raw_hash):
    clocks={r['ACCESSION_NUMBER']:r for r in clocks};rows=[];blocked=[]
    for record in selected.to_dict('records'):
        accession=record['ACCESSION_NUMBER'];clock=clocks.get(accession)
        if not clock:
            blocked.append({'accession':accession,'asset':record['asset'],'reason':'PUBLIC_DISSEMINATION_EVIDENCE_ABSENT'});continue
        report=pd.Timestamp(record['REPORT_DATE'])
        available=pd.Timestamp(clock['available_at'])
        if available.tzinfo is None or available<pd.Timestamp(report.date(),tz='UTC'):raise ValueError('N-PORT availability precedes report date')
        assets=_number(record['NET_ASSETS'],'NET_ASSETS')
        for i,month in enumerate(nport_reference_months(report),1):
            sales=_number(record[f'SALES_FLOW_MON{i}'],f'SALES_FLOW_MON{i}');redemptions=_number(record[f'REDEMPTION_FLOW_MON{i}'],f'REDEMPTION_FLOW_MON{i}');reinvest=_number(record[f'REINVESTMENT_FLOW_MON{i}'],f'REINVESTMENT_FLOW_MON{i}')
            external=sales-redemptions if np.isfinite(sales) and np.isfinite(redemptions) else np.nan
            rows.append({'asset':record['asset'],'series_id':None if pd.isna(record.get('SERIES_ID')) else record.get('SERIES_ID'),
                'series_name':None if pd.isna(record.get('SERIES_NAME')) else record.get('SERIES_NAME'),'cik':record['CIK'],
                'accession':accession,'report_date':str(report.date()),'reference_month':month,'reference_time':pd.Period(month,freq='M').end_time.tz_localize('UTC').isoformat(),
                'available_at':available.isoformat(),'filing_date':record['FILING_DATE'],'submission_type':record['SUB_TYPE'],
                'sales':sales,'redemptions':redemptions,'reinvestment':reinvest,'external_flow':external,'reported_end_net_assets':assets,
                'raw_hash':raw_hash,'pit_tier':'B','clock_evidence':clock['clock_evidence'],'clock_evidence_hash':clock['clock_evidence_hash'],
                'original_publication_qualified':False,'organic_flow_qualified':False})
    return pd.DataFrame(rows),blocked


def rows_to_events(rows):
    out=[]
    for record in rows.to_dict('records'):
        for field in ['sales','redemptions','reinvestment','external_flow','reported_end_net_assets']:
            out.append({'entity':record['asset'],'source':'nport','field':field,'value':record[field],'unit':'USD',
                'reference_time':record['reference_time'],'available_at':record['available_at'],'raw_hash':record['raw_hash'],'pit_tier':'B',
                'accession':record['accession'],'series_id':record['series_id'],'clock_evidence':record['clock_evidence'],
                'clock_evidence_hash':record['clock_evidence_hash'],'original_publication_qualified':False,'organic_flow_qualified':False})
    frame=pd.DataFrame(out)
    if len(frame):validate_events(frame)
    return frame


def holdings_to_events(holdings, selected, clocks, raw_hash):
    clocks={r['ACCESSION_NUMBER']:r for r in clocks};meta={r.ACCESSION_NUMBER:r for r in selected.itertuples(index=False)};out=[];blocked=[]
    for record in holdings.to_dict('records'):
        accession=record['ACCESSION_NUMBER'];clock=clocks.get(accession);source=meta.get(accession)
        if not clock or source is None:
            blocked.append({'accession':accession,'asset':record['asset'],'reason':'HOLDING_CLOCK_OR_METADATA_ABSENT'});continue
        reference=pd.Timestamp(source.REPORT_DATE)
        for field,unit in [('holding_count','count'),('top10_abs_percentage','percent_net_assets'),('gross_abs_percentage','percent_net_assets')]:
            out.append({'entity':record['asset'],'source':'nport','field':field,'value':record[field],'unit':unit,
                'reference_time':pd.Timestamp(reference.date(),tz='UTC').isoformat(),'available_at':clock['available_at'],'raw_hash':raw_hash,'pit_tier':'B',
                'accession':accession,'series_id':None if pd.isna(source.SERIES_ID) else source.SERIES_ID,'clock_evidence':clock['clock_evidence'],
                'clock_evidence_hash':clock['clock_evidence_hash'],'original_publication_qualified':False,'organic_flow_qualified':False})
    frame=pd.DataFrame(out)
    if len(frame):validate_events(frame)
    return frame,blocked


def acquire_bulk(repo,runtime,client_factory=Acquisition):
    repo,runtime=Path(repo),Path(runtime);plan,plan_path=load_plan(repo);validate_sec_user_agent(os.environ.get('SEC_USER_AGENT',''))
    plan_id=digest(plan);commit_bundle(runtime/'artifacts/nport_bulk_plans'/plan_id,{'plan.json':plan},{'reserved_access':False})
    discovery=client_factory(runtime,32*1024**2);html,_=discovery.html(INDEX['nport'])
    total=0;quarter_receipts=[];selected_frames=[];holding_frames=[]
    for quarter in plan['quarters']:
        url=discover_zip(html,'nport',quarter);client=client_factory(runtime,plan['max_bytes_per_quarter'])
        raw=client.cached(url,'nport_zip') or client.fetch(url,'nport_zip',metadata={'quarter':quarter,'reserved_access':False,'plan_id':plan_id})
        total+=raw['bytes']
        if total>plan['max_bytes_total']:raise RuntimeError('PAUSED_BUDGET: N-PORT total byte ceiling reached')
        target=target_accessions(raw['path'],plan,quarter);target['quarter_raw_hash']=raw['sha256'];selected_frames.append(target)
        holdings=target_holding_summaries(raw['path'],target);holdings['quarter']=quarter;holdings['quarter_raw_hash']=raw['sha256'];holding_frames.append(holdings)
        quarter_receipts.append({'quarter':quarter,'raw':raw,'target_accessions':len(target),'target_holding_summaries':len(holdings)})
    selected=pd.concat(selected_frames,ignore_index=True) if selected_frames else pd.DataFrame()
    if selected.empty:raise RuntimeError('BLOCKED_DATA: no registered target funds discovered in N-PORT bulk')
    clock_client=client_factory(runtime,512*1024**2);clocks=[]
    if selected.ACCESSION_NUMBER.nunique()>plan['max_accession_clock_requests']:raise RuntimeError('PAUSED_BUDGET: accession clock request ceiling reached')
    for record in selected[['ACCESSION_NUMBER','CIK']].drop_duplicates().to_dict('records'):
        url=accession_index_url(record['CIK'],record['ACCESSION_NUMBER']);raw=clock_client.cached(url,'html') or clock_client.fetch(url,'html',metadata={'accession':record['ACCESSION_NUMBER'],'reserved_access':False,'plan_id':plan_id})
        accepted=parse_accepted(Path(raw['path']).read_text(errors='replace'));available=accepted+pd.Timedelta(minutes=plan['availability_lag_minutes'])
        if available>=pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError('Reserved N-PORT public clock encountered')
        clocks.append({'ACCESSION_NUMBER':record['ACCESSION_NUMBER'],'accepted_at':accepted.isoformat(),'available_at':available.isoformat(),
            'pit_tier':'B','clock_evidence':url,'clock_evidence_hash':raw['sha256']})
    rows=[];blocked=[];holding_event_frames=[]
    holdings_all=pd.concat(holding_frames,ignore_index=True) if holding_frames else pd.DataFrame()
    for quarter,group in selected.groupby('quarter',sort=False):
        raw=next(x['raw'] for x in quarter_receipts if x['quarter']==quarter)
        canonical,b=canonical_rows(group,clocks,raw['sha256']);rows.append(canonical);blocked.extend(b)
        h=holdings_all[holdings_all.quarter.eq(quarter)] if len(holdings_all) else holdings_all
        he,hb=holdings_to_events(h,group,clocks,raw['sha256']);holding_event_frames.append(he);blocked.extend(hb)
    rows=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame();flow_events=rows_to_events(rows)
    holding_events=pd.concat(holding_event_frames,ignore_index=True) if holding_event_frames else pd.DataFrame()
    events=pd.concat([flow_events,holding_events],ignore_index=True) if len(holding_events) else flow_events
    payload=json.loads(events.sort_values(['available_at','entity','field','reference_time']).to_json(orient='records'))
    identity=digest({'plan_id':plan_id,'quarter_hashes':[x['raw']['sha256'] for x in quarter_receipts],'clock_hashes':sorted(x['clock_evidence_hash'] for x in clocks),'events':payload})
    folder=runtime/'artifacts/nport_canonical'/identity
    commit_bundle(folder,{'events.json':payload,'target_rows.json':json.loads(rows.to_json(orient='records')),
        'holding_summaries.json':json.loads(holdings_all.to_json(orient='records')) if len(holdings_all) else [],
        'clocks.json':clocks,'blocked.json':blocked},{'plan_id':plan_id,'pit_tier':'B','qualified_for_final':False})
    coverage={asset:{'accessions':int((selected.asset==asset).sum()),'quarters':sorted(selected.loc[selected.asset.eq(asset),'quarter'].unique().tolist()),
        'series_ids':sorted(str(v) for v in selected.loc[selected.asset.eq(asset),'SERIES_ID'].dropna().unique())} for asset in plan['targets']}
    result={'state':'SUCCEEDED_RECONSTRUCTED_NPORT_TARGET_FLOWS','artifact_id':identity,'plan_id':plan_id,
        'events_relative_path':str((folder/'events.json').relative_to(runtime)),'events_sha256':file_hash(folder/'events.json'),
        'rows':len(rows),'event_rows':len(events),'quarter_receipts':quarter_receipts,'coverage':coverage,'blocked':blocked,
        'non_applicable_assets':plan['non_applicable_assets'],'pit_tier':'B','original_publication_qualified':False,'organic_flow_qualified':False,
        'qualified_for_final':False,'reserved_access':False,'created_at':now(),'claim_boundary':plan['claim_boundary']}
    atomic_json(repo/'reports/nport_bulk_integration.json',result);return result
