"""Usable incremental CLI. Unsupported research phases fail closed with explicit evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .runtime import paths,atomic_json,now,file_hash,validate_bundle,digest
from .reporting import render,validate_lineage


def run_script(name):
    return subprocess.run([sys.executable,'scripts/'+name],check=False).returncode


def reconcile(repo,runtime):
    checked=[];invalid=[]
    for receipt in (runtime/'raw').glob('*/*.receipt.json'):
        try:
            r=json.loads(receipt.read_text())
            if file_hash(r['path'])!=r['sha256']:
                raise ValueError('Raw checksum mismatch')
            checked.append(str(receipt))
        except Exception as e:
            invalid.append({'path':str(receipt),'reason':str(e)})
    partial=[str(p) for p in (runtime/'raw').glob('*/*.partial')]
    return {'read_only':True,'raw_receipts_verified':len(checked),'invalid':invalid,'incomplete_partials':partial,
            'final_sealed':not (runtime/'ledger/final_access.json').exists()}


def processed_audit(repo):
    from .data import eia_events,auxiliary_rows
    path=repo/'reports/eia_development_acquisition.json'
    if not path.exists():
        return {'state':'BLOCKED_DATA','reason':'No EIA development archive manifest'}
    records=json.loads(path.read_text())['records']
    events=eia_events(records);rows=auxiliary_rows(records)
    eligible=rows[rows.y.notna()] if len(rows) else rows
    train=eligible[eligible.decision_time<'2018-01-01'] if len(eligible) else eligible
    result={'created_at':now(),'canonical_event_rows':len(events),'friday_origins':len(rows),'eligible_auxiliary_decisions':len(eligible),
            'missing_horizon_targets':len(rows)-len(eligible),
            'training_distinct_dates_before_2018':int(train.decision_time.nunique()) if len(train) else 0,
            'required_min_train_dates':156,'pit_tier':'B','tier_a_qualified':False,
            'target_status':'next_archived_issue_change_original_vintage_unverified',
            'state':'BLOCKED_DATA' if len(train)<156 else 'READY_FOR_ADDITIONAL_CLOCK_QA',
            'reason':'Complete chronological release panel and original-release qualification required before authoritative experiments',
            'reserved_access':False}
    result['target_eligibility']='Next issue calendar date must be strictly after Friday origin; conservative availability upper bounds cannot make a same-day public target appear future'
    atomic_json(repo/'reports/processed_data_audit.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('doctor');q.add_argument('--profile',default='local_4090')
    q=sub.add_parser('desk-study');q.add_argument('--verify-sources',action='store_true')
    q=sub.add_parser('reconcile');q.add_argument('--read-only',action='store_true',required=True)
    q=sub.add_parser('plan');q.add_argument('--mode',choices=['pilot','development'],default='pilot')
    q=sub.add_parser('pipeline');q.add_argument('--mode',choices=['pilot','development','locked'],required=True);q.add_argument('--resume',action='store_true');q.add_argument('--freeze-receipt');q.add_argument('--authorize-final-read',action='store_true')
    q=sub.add_parser('qualify');q.add_argument('--suite',choices=['all-local'],default='all-local')
    q=sub.add_parser('freeze');q.add_argument('--study',default='sgqx-v3')
    q=sub.add_parser('report');q.add_argument('--study',default='sgqx-v3');q.add_argument('--validate-lineage',action='store_true')
    q=sub.add_parser('forward-once');q.add_argument('--as-of',required=True)
    q=sub.add_parser('track-development');q.add_argument('--track',choices=['Main-A','Nested-B','both'],default='both');q.add_argument('--input-version',choices=['auto','base','v31','v311'],default='auto')
    a=p.parse_args(argv);repo,runtime=paths()
    if a.command=='track-development':
        return subprocess.run([sys.executable,'scripts/run_track_development.py','--track',a.track,'--input-version',a.input_version],check=False).returncode
    if a.command=='doctor':
        r=subprocess.run([sys.executable,'scripts/audit_runtime.py'],capture_output=True,text=True)
        if r.returncode:
            print(r.stderr,file=sys.stderr);return r.returncode
        audit=json.loads(r.stdout)
        audit['environment_id']=digest({'packages':audit['resolved_packages'],'dispatcher':audit['dispatcher_sha256'],
                                        'python':audit['python'],'gpu_inventory':audit['gpu_inventory'].split(',')[0:2]})
        atomic_json(repo/'reports/environment_audit.json',audit)
        print(json.dumps(audit,indent=2));return 0
    if a.command=='desk-study':
        rc=run_script('build_desktop_study.py')
        if rc:return rc
        return run_script('verify_source_docs.py') if a.verify_sources else 0
    if a.command=='reconcile':
        result=reconcile(repo,runtime);print(json.dumps(result,indent=2));return 1 if result['invalid'] else 0
    if a.command=='plan':
        from .planning import run_plan
        result=run_plan(repo,runtime,a.mode)
        atomic_json(repo/'reports/run_plan.json',result);print(json.dumps(result,indent=2));return 0
    if a.command=='pipeline':
        result=reconcile(repo,runtime)
        if result['invalid']:
            print(json.dumps(result,indent=2));return 1
        if a.mode=='locked':
            receipt=Path(a.freeze_receipt) if a.freeze_receipt else repo/'.local/freeze_receipt.json'
            if not receipt.exists():
                print('BLOCKED_DATA: qualified freeze receipt absent; final remains sealed',file=sys.stderr);return 2
            from .final import execute_frozen
            try:print(json.dumps(execute_frozen(repo,runtime,receipt),indent=2));return 0
            except (RuntimeError,PermissionError,FileNotFoundError) as exc:
                print(str(exc),file=sys.stderr);return 2
        rc=run_script('acquire_public_pilot.py' if a.mode=='pilot' else 'acquire_eia_remaining.py')
        print(json.dumps(processed_audit(repo),indent=2))
        if a.mode=='development':
            from .auxiliary import run
            try:run(repo,runtime)
            except (RuntimeError,PermissionError,FileNotFoundError) as exc:
                print(str(exc),file=sys.stderr)
        render(repo);return rc if rc else 2
    if a.command=='qualify':
        rc=run_script('run_validation.py')
        if rc:return rc
        return run_script('qualify_systems.py')
    if a.command=='freeze':
        from .integrity import create_freeze
        try:
            receipt=create_freeze(repo,runtime);print(json.dumps({'freeze_id':digest(receipt),'receipt_path':str(repo/'.local/freeze_receipt.json')}));return 0
        except (RuntimeError,PermissionError,FileNotFoundError) as exc:
            print(str(exc),file=sys.stderr);return 2
    if a.command=='report':
        if a.study!='sgqx-v3':raise ValueError('Unknown study')
        render(repo)
        if a.validate_lineage:validate_lineage(repo)
        print(str(repo/'reports/technical_report.md'));return 0
    if a.command=='forward-once':
        from .forward import produce_once
        try:print(json.dumps(produce_once(repo,runtime,a.as_of),indent=2));return 0
        except (RuntimeError,PermissionError,FileNotFoundError) as exc:
            print(str(exc),file=sys.stderr);return 2
    return 1


if __name__=='__main__':raise SystemExit(main())
