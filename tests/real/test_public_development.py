"""Genuine cached public data/artifact checks, distinct from synthetic tests."""
import json
import pandas as pd
from signalforge.runtime import paths,file_hash,validate_bundle
from signalforge.data import auxiliary_rows
from signalforge.auxiliary import sequences
from signalforge.sources import parse_cftc
from signalforge.positioning import positioning_features


def test_real_auxiliary_source_clock_and_target_grid():
    repo,runtime=paths();manifest=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
    assert len(manifest['records'])==manifest['required']
    assert all('/2024/' not in r['release_page'] for r in manifest['records'])
    for record in manifest['records']:
        if record['state']=='SUCCEEDED':assert file_hash(record['raw']['path'])==record['raw']['sha256']
    frame=auxiliary_rows(manifest['records']);eligible=frame[frame.y.notna()]
    assert len(eligible)>156
    assert (pd.to_datetime(eligible.label_end,utc=True)>pd.to_datetime(eligible.decision_time,utc=True)).all()
    assert (pd.to_datetime(frame.max_dependency_available_at,utc=True)<=pd.to_datetime(frame.decision_time,utc=True)).all()
    assert not (eligible.raw_hash==eligible.target_raw_hash).any()
    assert frame.target_status.eq('FAILED_SOURCE_TARGET').any()
    assert frame.target_status.eq('MISSING_NEXT_RELEASE_METADATA').any()
    assert frame.pit_tier.eq('B').all()


def test_real_pdf_correction_preserves_published_precision_and_bad_csv():
    repo,runtime=paths();manifest=json.loads((repo/'reports/eia_development_acquisition.json').read_text())
    record=next(r for r in manifest['records'] if '/2019_07_03/' in r['release_page'])
    assert record['raw']['source_url'].endswith('/pdf/table4.pdf')
    assert record['event']['published_precision']==.1
    assert record['event']['series']['Commercial (Excluding SPR)']['change']==-1.1
    assert record['superseded_source_attempt']['state']=='FAILED_SOURCE'
    assert file_hash(repo/record['visual_verification']['image'])==record['visual_verification']['sha256']
    validate_bundle(runtime/'artifacts/incidents/eia_2019_duplicate_original_inputs')


def test_real_cpu_reload_and_fixed_outage_receipts():
    repo,runtime=paths();reload=json.loads((repo/'reports/auxiliary_real_reload_qualification.json').read_text())
    assert reload['n_verified_outer_bundles']==75
    for check in reload['checks']:
        validate_bundle(runtime/'artifacts/auxiliary'/check['run_id'])
        assert max(check['max_mean_abs_delta'],check['max_quantile_abs_delta'])<=1e-10
    robustness=json.loads((repo/'reports/auxiliary_cpu_robustness.json').read_text())
    assert robustness['no_refitting_or_model_seed_selection'] and not robustness['final_access']
    assert robustness['n_market_dates']==260 and len(robustness['comparison_table'])==10
    assert all(r['paired']['n_training_seeds_are_independent_market_samples'] is False for r in robustness['comparison_table'])


def test_real_cftc_archives_and_positioning_semantics():
    repo,runtime=paths();manifest=json.loads((repo/'reports/cftc_development_acquisition.json').read_text())
    assert manifest['completed']==manifest['required']==28
    for record in manifest['records']:
        assert record['year']<=2023 and file_hash(record['raw']['path'])==record['raw']['sha256']
        assert record['available_at'] is None and not record['model_qualified'] and not record['positions_are_cashflows']
        if record['year']==2010:
            frame=parse_cftc(record['raw']['path'],record['family']);features=positioning_features(frame,record['family'])
            assert len(features)==record['audit']['rows'] and features.open_interest.ge(0).all()


def test_real_complete_core_and_age_aware_baseline_receipts():
    repo,runtime=paths();core=json.loads((repo/'reports/auxiliary_development_results.json').read_text())
    assert core['state']=='SUCCEEDED_DIAGNOSTIC' and len(core['results'])==270
    reload=json.loads((repo/'reports/auxiliary_all_family_reload_qualification.json').read_text())
    assert reload['state']=='SUCCEEDED_ALL_FAMILY_RELOAD' and reload['n_verified_outer_bundles']==270
    assert len({r['family'] for r in core['results']})==18
    transforms={};grids={}
    for row in core['results']:
        directory=runtime/'artifacts/auxiliary'/row['run_id'];validate_bundle(directory)
        assert row['state']=='SUCCEEDED' and row['pit_tier']=='B'
        prediction=json.loads((directory/'predictions.json').read_text())
        transform=json.loads((directory/'transform.json').read_text())
        key=row['outer_year']
        assert transforms.setdefault(key,transform)==transform
        expected={'dates':prediction['decision_time'],'targets':prediction['y'],'normalizer':row['normalizer_id']}
        assert grids.setdefault(key,expected)==expected
        if row['family']=='lightgbm':
            assert row['feature_variant'] is None and len(transform['center'])==14
    assert sum(len(g['dates']) for g in grids.values())==260
    assert not core.get('qualified_for_final',False)
