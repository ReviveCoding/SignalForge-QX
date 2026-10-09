"""Separate fixed and observed N-PORT cohorts without survivor aggregation."""
import numpy as np
import pandas as pd
from .nport import month_snapshot


def cohort_audit(flows,months,decision,registered_cohort):
    decision=pd.Timestamp(decision)
    if decision.tzinfo is None:raise ValueError('Aware as-of decision required')
    if decision>=pd.Timestamp('2024-01-01T00:00Z'):raise PermissionError('Development cohort audit cannot expose reserved observations')
    months=list(months)
    if not months or months!=sorted(set(months)):raise ValueError('Unique chronological reference months required')
    if any(str(month)>='2024-01' for month in months):raise PermissionError('Reserved reference cohort denied')
    cohort=list(registered_cohort)
    if not cohort or len(set(cohort))!=len(cohort):raise ValueError('Fixed registered cohort required')
    known=flows[pd.to_datetime(flows.available_at,utc=True)<=decision].copy()
    previous=set();cards=[]
    for month in months:
        current=known[known.reference_month.eq(month)].copy()
        observed=set(current.series_id)
        fixed=month_snapshot(flows,month,decision,cohort)
        dynamic=month_snapshot(flows,month,decision,sorted(observed)) if observed else None
        rows=dynamic['rows'] if dynamic else current
        assets=pd.to_numeric(rows.reported_end_net_assets,errors='raise')
        if np.isinf(assets).any() or (assets.dropna()<0).any():raise ValueError('Invalid reported concentration amounts')
        reference_aligned=('report_date' in rows and rows.report_date.map(lambda value:pd.Timestamp(value).strftime('%Y-%m')).eq(month).all())
        complete_assets=bool(len(assets)) and reference_aligned and assets.notna().all() and assets.sum()>0
        cards.append({'reference_month':month,'fixed_state':fixed['state'],'fixed_coverage':fixed['coverage'],
            'fixed_missing_members':fixed['missing_members'],'fixed_external_flow':fixed['external_flow'],
            'observed_state':dynamic['state'] if dynamic else 'BLOCKED_DATA','observed_entities':sorted(observed),
            'observed_external_flow':dynamic['external_flow'] if dynamic else None,
            'entered_since_previous_observed_month':sorted(observed-previous),
            'exited_since_previous_observed_month':sorted(previous-observed),
            'known_raw_rows':len(current),'selected_entity_rows':len(rows),'known_revisions':len(current)-len(rows),
            'missing_flow_entities':int(rows.external_flow.isna().sum()),'true_zero_flow_entities':int(rows.external_flow.eq(0).sum()),
            'reported_assets_HHI':float(((assets/assets.sum())**2).sum()) if complete_assets else None,
            'concentration_complete':bool(complete_assets),'selected_accessions':rows.accession.tolist(),
            'concentration_reference_aligned':bool(reference_aligned),
            'dynamic_cohort_is_fixed_cohort':observed==set(cohort)})
        previous=observed
    return {'state':'SUCCEEDED_COHORT_SOFTWARE_AUDIT','asof':decision.isoformat(),'fixed_cohort':cohort,
        'months':cards,'organic_flow_qualified':False,'dynamic_survivor_total_qualifies_fixed_cohort':False,
        'rate_denominator_used':False,'reported_end_assets_used_for_concentration_only':True}
