"""Completed studies are terminal; a later code hash is not a new run license."""


def completed_full_receipt(full,admission,frozen_plan,plan_id):
    if not full or full.get('state')!='SUCCEEDED_SUCCESSOR_GPU_DEVELOPMENT':return False
    if not admission or admission.get('state')!='ADMITTED_MEASURED_SUCCESSOR_BILL':
        raise PermissionError('INTEGRITY_OR_HASH_GATE: completed study admission absent')
    if full.get('execution_plan_id')!=plan_id or admission.get('execution_plan_id')!=plan_id:
        raise PermissionError('INTEGRITY_OR_HASH_GATE: completed study plan differs')
    mode=full.get('plan_mode');plan=frozen_plan['plans'].get(mode)
    if not plan:raise PermissionError('INTEGRITY_OR_HASH_GATE: completed study mode absent')
    expected=plan['planned_fit_count']
    tracks=full.get('tracks',[]);rows=[row for track in tracks for row in track.get('results',[])]
    if (type(full.get('planned_fits')) is not int or full['planned_fits']!=expected or
        full.get('expected_outer_results')!=expected or len(rows)!=expected or
        {track.get('track') for track in tracks}!=set(frozen_plan['tracks']) or
        len(tracks)!=len(frozen_plan['tracks']) or len({r.get('run_id') for r in rows})!=expected or
        any(r.get('state')!='SUCCEEDED' for r in rows) or full.get('reserved_access') is not False):
        raise PermissionError('INTEGRITY_OR_HASH_GATE: incomplete/corrupt completed study receipt')
    # This decision suppresses duplicate dispatch; it never upgrades source/PIT/final qualification.
    return True
