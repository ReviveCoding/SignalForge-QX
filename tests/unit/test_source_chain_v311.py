import json
from pathlib import Path
from signalforge.source_requests import validate_fred_plan,FRED_SEGMENT_MAX_CALENDAR_DAYS

ROOT=Path(__file__).resolve().parents[2]

def test_segmented_fred_plan_is_deterministic_bounded_child():
    legacy=json.loads((ROOT/'.local/fred_request_plan.json').read_text(encoding='utf-8-sig'))
    plan=json.loads((ROOT/'.local/fred_request_plan_v31.json').read_text(encoding='utf-8-sig'))
    validate_fred_plan(plan,require_json_safe_segments=True)
    assert {r['series_id'] for r in plan['series']}=={r['series_id'] for r in legacy['series']}
    assert len(plan['series'])==20
    for series in {r['series_id'] for r in plan['series']}:
        rows=sorted([r for r in plan['series'] if r['series_id']==series],key=lambda r:r['realtime_start'])
        assert [r['output_type'] for r in rows]==[1,3,3,3]
        for r in rows[1:]:
            import pandas as pd
            assert (pd.Timestamp(r['realtime_end'])-pd.Timestamp(r['realtime_start'])).days+1<=FRED_SEGMENT_MAX_CALENDAR_DAYS

def test_v311_feature_contract_changes_only_indpro_semantics():
    old=json.loads((ROOT/'configs/track_feature_specs_v31.json').read_text())
    new=json.loads((ROOT/'configs/track_feature_specs_v311.json').read_text())
    def by_name(cfg,source):
        return {x['entity']:x for x in cfg['sources'][source]}
    old_macro=by_name(old,'macro');new_macro=by_name(new,'macro')
    assert set(old_macro)==set(new_macro)=={'CPIAUCSL','INDPRO','UNRATE','DGS2','DGS10'}
    for entity in set(old_macro)-{'INDPRO'}:
        assert old_macro[entity]==new_macro[entity]
    assert new_macro['INDPRO']['field']=='yoy_percent'
    assert new_macro['INDPRO']['unit']=='percent_yoy'
    for source in ['cftc','eia','nport']:
        assert old['sources'][source]==new['sources'][source]

def test_resume_chain_opens_main_before_nport_and_refreshes_after_nport():
    text=(ROOT/'scripts/Resume-SignalForge-Completion.ps1').read_text()
    register=text.index('register_fred_segmented_plan.py')
    acquire=text.index('acquire_keyed_sources.py')
    macro=text.index('build_macro_canonical_integration.py')
    first_build=text.index('build_track_source_inputs_v311.py')
    first_cpu=text.index("track-development','--track','both','--input-version','auto")
    nport=text.index('build_nport_bulk_integration.py')
    reclock=text.index('reclock_nport_v311.py')
    second_build=text.index('build_track_source_inputs_v311.py',first_build+1)
    second_cpu=text.index("track-development','--track','both','--input-version','v311")
    assert register<acquire<macro<first_build<first_cpu<nport<reclock<second_build<second_cpu

def test_indpro_same_vintage_yoy_is_rebase_invariant():
    from signalforge.macro_integration import _indpro_yoy
    rows=[
        {'series_id':'INDPRO','reference_time':'2019-01-31T00:00:00+00:00','realtime_start':'2020-01-15','available_at':'2020-01-16T12:00:00+00:00','value':100.0,'raw_hash':'a'*64},
        {'series_id':'INDPRO','reference_time':'2020-01-31T00:00:00+00:00','realtime_start':'2020-01-15','available_at':'2020-01-16T12:00:00+00:00','value':110.0,'raw_hash':'b'*64},
        {'series_id':'INDPRO','reference_time':'2019-01-31T00:00:00+00:00','realtime_start':'2020-02-15','available_at':'2020-02-16T12:00:00+00:00','value':200.0,'raw_hash':'c'*64},
        {'series_id':'INDPRO','reference_time':'2020-01-31T00:00:00+00:00','realtime_start':'2020-02-15','available_at':'2020-02-16T12:00:00+00:00','value':220.0,'raw_hash':'d'*64},
    ]
    events=_indpro_yoy(rows)
    values=[e['value'] for e in events if e['reference_time']=='2020-01-31T00:00:00+00:00']
    assert len(values)==2
    assert all(abs(v-10.0)<1e-12 for v in values)

def test_nport_conservative_clock_skips_weekend_and_federal_holiday():
    import pandas as pd
    import pytest
    from signalforge.nport_reclock import conservative_development_clock
    # Friday before Labor Day 2023 -> Tuesday Sep 5, never Saturday/Sunday/Monday holiday.
    t=conservative_development_clock(pd.Timestamp('2023-09-01T18:00:00',tz='America/New_York'))
    assert t.isoformat()=='2023-09-05T23:59:59-04:00'
    with pytest.raises(ValueError,match='Aware'):
        conservative_development_clock(pd.Timestamp('2023-09-01T18:00:00'))

def test_cli_forwards_track_input_version(monkeypatch,tmp_path):
    import signalforge.cli as cli
    seen={}
    class Result:
        returncode=0
    def fake_run(args,check=False,**kwargs):
        seen['args']=args
        return Result()
    monkeypatch.setattr(cli,'paths',lambda:(tmp_path,tmp_path))
    monkeypatch.setattr(cli.subprocess,'run',fake_run)
    assert cli.main(['track-development','--track','Main-A','--input-version','v311'])==0
    assert seen['args'][-2:]==['--input-version','v311']

def test_stable_macro_levels_refuse_silent_unit_relabeling():
    import pytest
    from signalforge.macro_integration import _stable_level_event
    row={'series_id':'DGS10','unit':'Basis Points','reference_time':'2020-01-01T00:00:00+00:00',
         'available_at':'2020-01-02T12:00:00+00:00','value':1.5,'raw_hash':'a'*64,
         'realtime_start':'2020-01-01'}
    with pytest.raises(ValueError,match='unit changed'):
        _stable_level_event(row)

def test_single_track_runner_does_not_own_canonical_qualification_contract():
    text=(ROOT/'scripts/run_track_development.py').read_text()
    assert "if a.track=='both':" in text
    assert "track_input_qualification_'+version+'_'+safe" in text


def test_cpu_track_runner_does_not_consume_parent_gpu_seconds():
    text=(ROOT/'scripts/run_track_development.py').read_text()
    assert 'CPU_FAMILIES' in text
    assert 'budget=None' in text
    assert 'development_compute.sqlite' not in text
    assert 'ComputeBudget' not in text


def test_experiment_status_releases_stale_main_nested_blocker_when_cpu_ready():
    text=(ROOT/'scripts/update_experiment_status.py').read_text()
    assert 'track_cpu_ready=all(' in text
    assert "cards['E01'].update(state='PARTIAL'" in text
    assert "cards['E02'].update(state='PARTIAL'" in text


def test_gpu_queue_requires_validation_newer_than_v311_qualification():
    text=(ROOT/'scripts/gpu_queue_status.py').read_text()
    assert 'newer_track_qualification' in text
    assert 'qualification.stat().st_mtime_ns>test_path.stat().st_mtime_ns' in text
    assert "validation['passed']=False" in text


def test_gpu_queue_retries_preserved_blocker_only_after_verified_repair():
    text=(ROOT/'scripts/gpu_queue_status.py').read_text()
    assert "resolved_repair=bool(incident and incident.get('state')=='RESOLVED_VERIFIED')" in text
    assert "retryable=bool(receipt and receipt.get('state')!='SUCCEEDED_MEASURED_SUCCESSOR_PILOT' and resolved_repair)" in text
    assert "retry_full=bool(full_doc and full_doc.get('state')=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE' and resolved_repair)" in text
    assert "admission.get('state')=='SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE' and resolved_repair" in text


def test_recovery_status_supersedes_only_the_verified_manual_block_checkpoint():
    text=(ROOT/'scripts/successor_recovery_status.py').read_text()
    assert "terminal.get('state')=='MANUAL_BLOCKED_ADDITIONAL_RUNTIME_REPAIR_AUTHORIZATION'" in text
    assert "incident.get('state')=='RESOLVED_VERIFIED'" in text
    assert 'superseded_manual_block' in text


def test_macro_refuses_unbound_mutable_acquisition_report(tmp_path):
    import pytest
    from signalforge.macro_integration import build_macro_integration
    from signalforge.runtime import atomic_json,commit_bundle,digest
    repo=tmp_path/'repo';runtime=tmp_path/'runtime';(repo/'reports').mkdir(parents=True)
    acquisition={'state':'SUCCEEDED_RAW_FRED_DEVELOPMENT_VINTAGES','records':[],'raw_receipts':[]}
    identity=digest(acquisition)
    commit_bundle(runtime/'artifacts/fred_request_results'/identity,{'results.json':acquisition},{})
    atomic_json(repo/'reports/fred_development_acquisition.json',{**acquisition,'artifact_id':identity,'plan_id':'changed'})
    with pytest.raises(ValueError,match='differs from immutable'):
        build_macro_integration(repo,runtime)


def test_secure_macro_resume_cleans_key_after_watcher_stop_failure():
    text=(ROOT/'scripts/Resume-SignalForge-Macro.ps1').read_text()
    cleanup=text.index("[Environment]::SetEnvironmentVariable('FRED_API_KEY',$null,'Process')")
    outer_try=text.rfind('try {',0,text.index('# Prevent the background watcher'))
    watcher_error=text.index("if($watcher){throw")
    finally_start=text.rfind('} finally {',0,cleanup)
    assert outer_try>=0 and outer_try<watcher_error<finally_start<cleanup
