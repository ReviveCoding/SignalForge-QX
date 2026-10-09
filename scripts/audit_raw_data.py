"""Deterministic raw QA, independent of targets; never deletes stressed observations."""
import json
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,file_hash,now
from signalforge.sources import parse_cftc

repo,runtime=paths();manifest=json.loads((repo/'reports/public_pilot.json').read_text());reports=[]
for source in manifest['sources']:
    if source['source']!='cftc' or 'raw' not in source:continue
    raw=source['raw']
    if file_hash(raw['path'])!=raw['sha256']:raise ValueError('Raw artifact changed')
    frame=parse_cftc(raw['path'],source['family']);fields={}
    for name in frame:
        if any(marker in name for marker in ['Positions_Long_All','Positions_Short_All','Open_Interest_All']):
            values=pd.to_numeric(frame[name],errors='coerce')
            observed=values.dropna()
            fields[name]={'missing_or_non_numeric':int(values.isna().sum()),'exact_zero':int(values.eq(0).sum()),
                          'negative':int(values.lt(0).sum()),'min':float(observed.min()),'max':float(observed.max()),
                          'q01':float(observed.quantile(.01)),'median':float(observed.median()),'q99':float(observed.quantile(.99))}
    counts=frame.groupby('report_date').CFTC_Contract_Market_Code.nunique()
    reports.append({'source':'cftc','family':source['family'],'raw_hash':raw['sha256'],'rows':len(frame),
                    'entities':int(frame.CFTC_Contract_Market_Code.nunique()),'report_dates':int(frame.report_date.nunique()),
                    'duplicate_contract_date_rows':int(frame.duplicated(['CFTC_Contract_Market_Code','report_date']).sum()),
                    'entities_per_date_min':int(counts.min()),'entities_per_date_max':int(counts.max()),
                    'contract_unit_labels':sorted(frame.Contract_Units.dropna().unique().tolist()),
                    'numeric_audit':fields,'tail_rows_deleted':0,'release_lag':'NOT_QUALIFIED: actual release evidence absent',
                    'revision_count':'NOT_QUALIFIED: only one currently retrieved annual vintage'})
eia_path=repo/'reports/eia_development_acquisition.json'
if eia_path.exists():
    eia=json.loads(eia_path.read_text());records=[r for r in eia['records'] if r['state']=='SUCCEEDED']
    dates=pd.to_datetime([r['event']['reference_time'] for r in records],utc=True)
    available=pd.to_datetime([r['event']['available_at'] for r in records],utc=True)
    values=np.array([r['event']['series']['Commercial (Excluding SPR)']['change'] for r in records])
    reports.append({'source':'eia','issues_parsed':len(records),'issues_required':eia['required'],'state':eia['state'],
                    'distinct_reference_dates':len(dates.unique()),'duplicate_references':len(dates)-len(dates.unique()),
                    'conservative_availability_lag_days_min':float(((available-dates).total_seconds()/86400).min()),
                    'conservative_availability_lag_days_max':float(((available-dates).total_seconds()/86400).max()),
                    'commercial_change_exact_zeros':int((values==0).sum()),'commercial_change_missing':int((~np.isfinite(values)).sum()),
                    'commercial_change_min':float(values.min()),'commercial_change_max':float(values.max()),
                    'unit':'million_barrels','unit_evidence':'reports/eia_unit_qualification.json','tail_rows_deleted':0,
                    'definition_break_audit':'2022-01-12 PDF footnote: lease stocks excluded from commercial crude week ended 2016-10-07; historical known-at not authenticated',
                    'first_publication_original_version':'UNVERIFIED','reserved_rows_inspected':0})
atomic_json(repo/'reports/raw_data_audit.json',{'created_at':now(),'sources':reports,'research_target_analysis':False})
print(json.dumps({'audited_sources':len(reports),'raw_data_audit':str(repo/'reports/raw_data_audit.json')},indent=2))
