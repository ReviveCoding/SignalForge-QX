"""Read-only forecast/freeze probe. Creates only a new prospective plan receipt."""
import json,sys
from pathlib import Path
from signalforge.runtime import digest,file_hash,atomic_json,now
from p1_pit_readiness_v34.gates import future_probe
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');O=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX');D=R/'reports/p1_pit_v34';T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering')
def probe():
    c=R/'reports/calibration_v34/calibration_frozen.json';bound=c.exists();cal=json.loads(c.read_text()) if bound else None
    flags={root.name:{'scientific_freeze_present':(root/'.local/freeze_receipt.json').exists(),'operational_freeze_present':(root/'.local/operational_freeze.json').exists(),'scientific_authorization_present':(root/'.local/scientific_authorization.json').exists()} for root in [O,R]}
    # Presence is not verification or permission; this lane never executes a freeze or forecast.
    out=future_probe(model_bound=bound,clock_qualified=False,cohort_dates=0);out['checked_at']=now();out['actual_file_presence']=flags;out['one_batch_final_state']='SEALED_MISSING_VERIFIED_FREEZE_AND_STRICT_PIT';out['authorization_read_only_presence_not_scope_override']=True
    if bound:
        sia=json.loads((R/'reports/v33_minimal_study_v2.json').read_text())['results'];plan={'study_id':'sgqx-v34-prospective-calibrated-SIA-readiness-v1','candidate':'fixed SIA v2 models plus train-only mature SIA OOF calibration v34; NOT promoted','calibration_id':cal['parameter_id'],'calibration_sha256':file_hash(c),'historical_candidate_bundles':[{'relative_bundle':s['relative_bundle'],'track':s['track'],'year':s['year'],'seed':s['seed'],'receipt_sha256':file_hash(T/s['relative_bundle']/'receipt.json'),'model_sha256':file_hash(T/s['relative_bundle']/'model.pt'),'prediction_sha256':file_hash(T/s['relative_bundle']/'predictions.npz')} for s in sia],'future_cohorts':{'per_track_untouched_blocks':2,'distinct_mature_dates_per_block':52,'assets':8,'independent_date_not_seed_unit':True},'seeds':[11,37,71],'metric':'date_equal_asset_equal_seed_average_normalized_pinball','comparison':'fixed strong statistical reference; shared train-only target scales; exact eligible grid','calibration_or_model_outcome_retuning':False,'requires':['verified operational freeze with actual timestamp','authenticated per-version first-public source clocks','per-cutoff mature labels and source-qualified forward inputs','independent governance','two actual future untouched 52-date cohorts'],'outcome_access':False,'training_authorized_by_this_plan':False,'reserved_access':False,'final_access':False,'scope':'evaluation design freeze only, not scientific/operational freeze; old 2020/2022 not independent evidence'};plan['plan_id']=digest(plan);path=R/'p1_pit_readiness_v34/future_evaluation_plan_v1.json'
        if path.exists():assert json.loads(path.read_text())==plan
        else:atomic_json(path,plan)
        out['future_evaluation_plan_id']=plan['plan_id']
    atomic_json(D/'future_readiness_current.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':probe()