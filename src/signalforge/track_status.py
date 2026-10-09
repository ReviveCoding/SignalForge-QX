"""Receipt-derived branch state; development completion never implies final gates."""
import json
from pathlib import Path


def continuation_phase_state(repo,phase,state,reason):
    terminal=Path(repo)/'reports/successor_continuation_terminal.json'
    if not terminal.exists():return state,reason
    receipt=json.loads(terminal.read_text())
    # Historical runtime blocks remain evidence, but cannot override later
    # completed development receipts. This never admits a final/freeze phase.
    def read(name):
        path=Path(repo)/'reports'/name
        return json.loads(path.read_text()) if path.exists() else {}
    if phase in {'P08','P09'}:
        full=read('successor_gpu_full.json');tracks={t['track']:t for t in full.get('tracks',[])}
        complete=(full.get('state')=='SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT' and
                  full.get('plan_mode')=='fixed_setting_development' and
                  full.get('completed_fit_calls')==full.get('expected_outer_results')==540 and
                  full.get('reserved_access') is False and full.get('qualified_for_final') is False and
                  set(tracks)=={'Main-A','Nested-B'} and
                  all(t.get('state')=='SUCCEEDED_DEVELOPMENT_SOFTWARE' and not t.get('blocked_folds') and
                      len(t.get('results',[]))==count and all(r.get('state')=='SUCCEEDED' for r in t['results'])
                      for name,count in [('Main-A',360),('Nested-B',180)] for t in [tracks[name]]))
        if complete:
            if phase=='P08':
                return 'COMPLETED_FIXED_SETTING_DEVELOPMENT_TIER_B','Both registered fixed-setting GPU tracks completed; no tuned-study or final qualification'
            return 'COMPLETED_FIXED_GRID_WITH_REGISTERED_GAPS','Fixed-grid development completed; retrained ablations, frozen outage interventions and H2/H3 remain unexecuted'
    if phase=='P10':
        calibration=[read('calibration_'+t+'.json') for t in ['Main_A','Nested_B']]
        if all(c.get('state')=='COMPLETED_DEVELOPMENT_CALIBRATION_TIER_B' and c.get('candidates') and
               c.get('qualified_for_final') is False for c in calibration):
            return 'COMPLETED_DEVELOPMENT_CALIBRATION_TIER_B','Track-specific development ensembles calibrated; unsupported tails retain identity; P1 and final gates remain blocked'
    if phase=='P07' and receipt.get('branches',{}).get('cpu_development')=='COMPLETED':
        return 'COMPLETED_DEVELOPMENT_TIER_B','Both source-extended CPU tracks completed; reconstructed development only'
    if phase=='P14':
        return 'COMPLETED_INCREMENTAL_DEVELOPMENT_REPORT','Incremental report/evidence executable; strict final research completion remains blocked'
    return 'SCIENTIFICALLY_BLOCKED_WITH_EVIDENCE','Independent completed branches retained; runtime/current-validation, strict source, Auxiliary consistency, P1 and freeze gates: reports/successor_continuation_terminal.json'


def track_phase_states(repo,phase):
    repo=Path(repo)
    def read(name):
        p=repo/'reports'/name
        return json.loads(p.read_text()) if p.exists() else {}
    qualification=read('track_input_qualification_v311.json')
    rows={r['track']:r for r in qualification.get('tracks',[])};out={}
    for track in ['Main-A','Nested-B']:
        row=rows.get(track,{})
        if phase in {'P12','P13'}:state='BLOCKED_FREEZE_GATES'
        elif phase=='P15':state='BLOCKED_OPERATIONAL_FREEZE'
        elif phase=='P10':state=read('calibration_'+track.replace('-','_')+'.json').get('state','NOT_FIT')
        elif phase in {'P08','P09'}:
            gpu=read('successor_gpu_full_'+track.replace('-','_')+'.json')
            state=gpu.get('state','CPU_SUCCEEDED_GPU_PENDING' if row.get('cpu_development',{}).get('state')=='SUCCEEDED_DEVELOPMENT_SOFTWARE' else row.get('state','BLOCKED_INPUT_QUALIFICATION'))
        elif phase in {'P05','P06'}:state=row.get('state','BLOCKED_INPUT_QUALIFICATION')
        else:state=row.get('cpu_development',{}).get('state',row.get('state','BLOCKED_INPUT_QUALIFICATION'))
        out[track]=state
    return out
