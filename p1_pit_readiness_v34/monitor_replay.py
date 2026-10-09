"""Per-date causal feature-clock monitoring replay; never live/backdated operation."""
import json
from pathlib import Path
import pandas as pd
from signalforge.runtime import atomic_json,now
from signalforge.panels import track_folds
from build_v33_readiness import load_original
from calibration_v34.runner import R,O,cfg
from calibration_v34.api import C,monitor

def main():
    cfg();cal=json.loads((O/'calibration_frozen.json').read_text());f=pd.read_csv(O/'outer_forecasts.csv');f=f[f.model=='SIA_calibrated_v34'].copy();f.decision_time=pd.to_datetime(f.decision_time,utc=True);rows=[]
    for track,year,info in [('Main-A',2020,'I3'),('Nested-B',2022,'I4')]:
        manifest,settings,panels,contexts,source=load_original(track);test=track_folds(panels[info],settings,track)[0][year]['test'];test=test[['decision_time','asset','max_dependency_available_at']].copy();test.decision_time=pd.to_datetime(test.decision_time,utc=True);g=f[f.track==track].merge(test,on=['decision_time','asset'],validate='many_to_one');assert len(g)==1248
        for (date,seed),h in g.groupby(['decision_time','seed']):
            at=pd.to_datetime(h.max_dependency_available_at,utc=True).max();assert at<=date;row=monitor(h[C].to_numpy(),h.scale.to_numpy(),cal['thresholds'][track],known_at=at,decision_time=date);rows.append({'track':track,'date':date.isoformat(),'seed':int(seed),'max_feature_dependency_at':at.isoformat(),'causal_calibration_fit_cutoff':cal['parameters'][track][str(seed)]['cutoff'],'actual_computation_is_now_not_historical':True,**row})
    pd.DataFrame(rows).to_csv(O/'monitor_per_date_clock_replay.csv',index=False);counts=pd.DataFrame(rows).groupby(['track','state']).size().to_dict();atomic_json(O/'monitor_clock_validation.json',{'passed':True,'replayed_date_seed_groups':len(rows),'prediction_rows':2496,'states':{str(k):int(v) for k,v in counts.items()},'all_feature_dependencies_at_or_before_origin':True,'clock_tier':'B reconstructed; not authentic first-public','actual_computed_at':now(),'not_backdated_forecasts':True,'thresholds_operationally_frozen':False,'live_notifications':0,'historical_batch_dry_run_is_not_online_clock_qualification':True,'rows':rows});print(json.dumps({'monitor_groups':len(rows),'rows':2496,'clock_check':'PASS_TIER_B_RETROSPECTIVE_ONLY','live_notifications':0}),flush=True)
if __name__=='__main__':main()