"""N-PORT submitted flows with independently evidenced dissemination clocks.

Bulk quarter, filing date and fiscal year-end never determine reference months
or public availability. This parser preserves missing values and amendments.
"""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .sources import validate_payload, nport_reference_months
from .pit import nport_external_flow
from .runtime import file_hash


FLOW_FIELDS = ['SALES_FLOW', 'REDEMPTION_FLOW', 'REINVESTMENT_FLOW']


def parse_nport(path, dissemination, evidence_registry=None):
    """Read two official TSV tables, joining accessions one to one.

    dissemination supplies accession, aware available_at, pit_tier and clock
    evidence. A additionally requires original_vintage_hash/public_evidence_hash.
    No timing evidence means no usable row, rather than a fabricated timestamp.
    """
    validate_payload(path, 'zip')
    tables = {}
    with zipfile.ZipFile(path) as archive:
        for table in ['SUBMISSION', 'FUND_REPORTED_INFO']:
            members = [m for m in archive.namelist()
                       if Path(m).name.upper() in {table+'.TSV', table+'.TXT'}]
            if len(members) != 1:
                raise ValueError('Missing or ambiguous N-PORT table: '+table)
            with archive.open(members[0]) as stream:
                tables[table] = pd.read_csv(stream, sep='\t', dtype=str, encoding='utf-8')
    submission, fund = tables['SUBMISSION'], tables['FUND_REPORTED_INFO']
    required_submission = {'ACCESSION_NUMBER', 'REPORT_DATE', 'FILING_DATE', 'SUB_TYPE'}
    required_fund = {'ACCESSION_NUMBER', 'SERIES_ID', 'NET_ASSETS'} | {
        f'{field}_MON{i}' for field in FLOW_FIELDS for i in range(1, 4)}
    if not required_submission <= set(submission) or not required_fund <= set(fund):
        raise ValueError('N-PORT flow schema changed')
    for table in [submission, fund]:
        if table.ACCESSION_NUMBER.isna().any() or table.ACCESSION_NUMBER.duplicated().any():
            raise ValueError('Missing or duplicate accession key')
    # Notice filings legitimately have no fund table, but orphan fund rows fail.
    if not set(fund.ACCESSION_NUMBER) <= set(submission.ACCESSION_NUMBER):
        raise ValueError('Orphan fund accession')
    joined = fund.merge(submission, on='ACCESSION_NUMBER', validate='one_to_one')
    clocks = pd.DataFrame(dissemination).copy()
    required_clocks = {'ACCESSION_NUMBER', 'available_at', 'pit_tier', 'clock_evidence'}
    if not required_clocks <= set(clocks) or clocks.ACCESSION_NUMBER.duplicated().any():
        raise ValueError('Unique independent dissemination evidence required')
    if not clocks.pit_tier.isin(['A', 'B']).all():
        raise ValueError('Usable N-PORT dissemination requires A or B evidence')
    for clock in clocks.to_dict('records'):
        if pd.Timestamp(clock['available_at']).tzinfo is None or not clock['clock_evidence']:
            raise ValueError('Aware public availability and evidence required')
        if clock['pit_tier'] == 'A' and not all(clock.get(k) for k in ['original_vintage_hash', 'public_evidence_hash']):
            raise ValueError('Tier A requires original version and public timing evidence')
        if clock['pit_tier']=='A':
            from .sources import allowed_url
            for key in ['original_vintage_hash','public_evidence_hash']:
                proof=(evidence_registry or {}).get(clock[key],{})
                if (proof.get('qualification_state')!='QUALIFIED_ORIGINAL_PUBLICATION' or
                    not allowed_url(proof.get('source_url','')) or not proof.get('path') or
                    file_hash(proof['path'])!=clock[key]):
                    raise ValueError('Tier A requires verified original/public evidence, not hash labels')
                known=pd.Timestamp(proof['public_available_at'])
                if known.tzinfo is None or known>pd.Timestamp(clock['available_at']):
                    raise ValueError('Tier A dissemination clock precedes public evidence')
    joined = joined.merge(clocks, on='ACCESSION_NUMBER', how='left', validate='one_to_one')
    raw_hash = file_hash(path)
    rows = []
    unqualified = []
    for record in joined.to_dict('records'):
        accession = record['ACCESSION_NUMBER']
        if pd.isna(record['available_at']):
            unqualified.append({'accession': accession, 'reason': 'PUBLIC_DISSEMINATION_EVIDENCE_ABSENT'})
            continue
        if record['SUB_TYPE'] not in {'NPORT-P', 'NPORT-P/A'} or pd.isna(record['SERIES_ID']):
            raise ValueError('Unexpected public fund type or missing series')
        report = pd.Timestamp(record['REPORT_DATE'])
        available = pd.Timestamp(record['available_at']).tz_convert('UTC')
        if pd.isna(report) or report != report.normalize() or not report.is_month_end:
            raise ValueError('Actual report date must be a month end')
        if available < report.tz_localize('UTC'):
            raise ValueError('Dissemination cannot precede reported period')
        net_assets = pd.to_numeric(record['NET_ASSETS'], errors='raise')
        if np.isinf(net_assets):
            raise ValueError('Infinite assets')
        for i, month in enumerate(nport_reference_months(report), 1):
            values = [pd.to_numeric(record[f'{field}_MON{i}'], errors='raise') for field in FLOW_FIELDS]
            sales, redemptions, reinvestment = values
            flow = nport_external_flow([sales], [redemptions])[0]
            if np.isinf(reinvestment) or reinvestment < 0:
                raise ValueError('Invalid reinvestment component')
            rows.append({'series_id': record['SERIES_ID'], 'reference_month': month,
                         'report_date': report.isoformat(), 'accession': accession,
                         'filing_date': record['FILING_DATE'], 'submission_type': record['SUB_TYPE'],
                         'available_at': available.isoformat(), 'pit_tier': record['pit_tier'],
                         'clock_evidence': record['clock_evidence'], 'raw_hash': raw_hash,
                         'original_vintage_hash':record.get('original_vintage_hash'),
                         'public_evidence_hash':record.get('public_evidence_hash'),
                         'sales': sales, 'redemptions': redemptions, 'reinvestment': reinvestment,
                         'external_flow': flow, 'reported_end_net_assets': net_assets,
                         'unit': 'USD', 'organic_flow_qualified': False,
                         'semantic_flags': ['OMNIBUS_NET_POSSIBLE', 'MERGER_LIQUIDATION_POSSIBLE']})
    return pd.DataFrame(rows), unqualified


def month_snapshot(flows, month, decision, cohort):
    """Fixed reference cohort and month, using only amendments already public.

    Missing members/components prevent an aggregate; no survivor-only subtotal
    is presented as the fixed-cohort aggregate. Tied versions are ambiguous.
    """
    decision = pd.Timestamp(decision)
    if decision.tzinfo is None:
        raise ValueError('Decision must be aware')
    cohort = list(cohort)
    if not cohort or len(set(cohort)) != len(cohort):
        raise ValueError('Unique fixed cohort required')
    rows = flows[(flows.reference_month == str(month)) & flows.series_id.isin(cohort)].copy()
    rows['available_at'] = pd.to_datetime(rows.available_at, utc=True)
    rows = rows[rows.available_at <= decision]
    if rows.duplicated(['series_id', 'available_at']).any():
        raise ValueError('Ambiguous amendment chronology')
    rows = rows.sort_values('available_at').drop_duplicates('series_id', keep='last')
    missing = sorted(set(cohort)-set(rows.series_id))
    complete = not missing and rows.external_flow.notna().all()
    return {'state': 'SUCCEEDED' if complete else 'BLOCKED_DATA', 'reference_month': str(month),
            'missing_members': missing, 'coverage': len(rows)/len(cohort),
            'external_flow': float(rows.external_flow.sum()) if complete else None,
            'reinvestment': float(rows.reinvestment.sum()) if complete and rows.reinvestment.notna().all() else None,
            'accessions': rows.accession.tolist(), 'rows': rows}
