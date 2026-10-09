"""Independent scalar reliability recomputation and preservation handoff. No fits."""
import os,json,sys,sqlite3,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from signalforge.runtime import digest,file_hash,code_hash,atomic_json,validate_bundle,commit_bundle,now
from calibration_v34.api import Q,C
from calibration_v34.runner import cfg,train_fit,R,T,O,source_hash
from p1_pit_readiness_v34.future_probe import probe
from diagnostics_v3.audit import preserve

def scalar(g):
    rows=[]
    for r in g.itertuples():
        q=np.array([getattr(r,col) for col in C]);y=r.target;s=r.scale;e=y-q;pin=np.array([max(float(t)*float(z),(float(t)-1)*float(z)) for t,z in zip(Q,e)])
        row={'date':r.decision_time,'asset':r.asset,'loss':pin.mean()/s,'raw_loss':pin.mean(),'residual':y-q[2],'abs':abs(y-q[2]),'normabs':abs(y-q[2])/s,'maxabs':abs(q).max(),'catastrophic':int(abs(q).max()>100),'crossing':int((np.diff(q)<0).sum())}
        for j,col in enumerate(C):row['coverage_'+col]=int(y<=q[j]);row['pin_'+col]=pin[j]/s;row['rawpin_'+col]=pin[j]
        for level,lo,hi in [(80,1,3),(90,0,4)]:
            alpha=1-level/100;width=q[hi]-q[lo];row['coverage'+str(level)]=int(q[lo]<=y<=q[hi]);row['width'+str(level)]=width;row['normwidth'+str(level)]=width/s;row['score'+str(level)]=(width+2/alpha*max(q[lo]-y,0)+2/alpha*max(y-q[hi],0))/s
        rows.append(row)
    a=pd.DataFrame(rows);date=a.groupby('date').mean(numeric_only=True);byasset=a.groupby('asset').loss.sum();m={'rows':len(a),'distinct_dates':len(date),'assets':g.asset.nunique(),'seeds':g.seed.nunique(),'normalized_pinball':float(date.loss.mean()),'raw_pinball':a.raw_loss.mean(),'coverage':[a['coverage_'+col].mean() for col in C],'pinball_by_quantile':[a['pin_'+col].mean() for col in C],'raw_pinball_by_quantile':[a['rawpin_'+col].mean() for col in C],'signed_median_residual':a.residual.mean(),'median_residual_median':a.residual.median(),'MAE':a['abs'].mean(),'MedAE':a['abs'].median(),'normalized_MAE':a.normabs.mean(),'error_p95':a['abs'].quantile(.95),'error_p99':a['abs'].quantile(.99),'top5_date_abs_error_fraction':date['abs'].nlargest(5).sum()/date['abs'].sum(),'loss_p95':a.loss.quantile(.95),'loss_p99':a.loss.quantile(.99),'max_abs_forecast':a.maxabs.max(),'p99_abs_forecast':a.maxabs.quantile(.99),'catastrophic_rows':a.catastrophic.sum(),'crossings':a.crossing.sum(),'top5_date_loss_fraction':date.loss.nlargest(5).sum()/date.loss.sum(),'asset_loss_fractions':(byasset/byasset.sum()).to_dict()}
    m['coverage_deviation_pp']=(100*(np.array(m['coverage'])-Q)).tolist()
    for level in (80,90):m.update({f'coverage{level}':a[f'coverage{level}'].mean(),f'width{level}':a[f'width{level}'].mean(),f'normalized_width{level}':a[f'normwidth{level}'].mean(),f'normalized_interval_score{level}':a[f'score{level}'].mean()})
    return m,a

def main():
    config=cfg();from calibration_v34.serialization_r1.adapter import identity_hash; repair=json.loads((O/'serialization_repair_v1r1.json').read_text()); assert identity_hash()==repair['revision']['revision_code_hash']; assert file_hash(O/'calibration_frozen.json')==repair['calibration_receipt_sha256']; val=json.loads((O/'independent_validation.json').read_text());assert val['passed'];ev=json.loads((O/'evaluation.json').read_text());f=pd.read_csv(O/'outer_forecasts.csv');checks=[]
    for (track,model),g in f.groupby(['track','model']):
        m,a=scalar(g);expected=ev['metrics'][track][model]
        for key,value in m.items():
            if isinstance(value,dict):
                for asset,x in value.items():assert np.isclose(x,expected[key][asset],rtol=0,atol=2e-12),(key,asset)
            else:assert np.allclose(value,expected[key],rtol=0,atol=2e-12),(track,model,key,value,expected[key])
        columns=['loss',*['coverage_'+col for col in C],'coverage80','coverage90','normwidth90','normabs'];rename={'normwidth90':'normalized_width90','normabs':'normalized_abs_error'};date=a.groupby('date')[columns].mean().sort_index();x=date.to_numpy()
        for block in (4,8,13):
            rng=np.random.default_rng(20261008+block);starts=rng.integers(0,len(x)-block+1,size=(2000,int(np.ceil(len(x)/block))));indices=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:len(x)];bounds=np.quantile(x[indices].mean(1),[.025,.975],axis=0)
            for j,col in enumerate(columns):assert np.allclose(bounds[:,j],expected['date_block_uncertainty'][str(block)]['columns'][rename.get(col,col)]['interval95'],rtol=0,atol=2e-12)
        checks.append({'track':track,'model':model,'scalar_reliability_metrics_checked':len(m),'bootstrap_blocks':[4,8,13],'rows':len(g)})
    oof=pd.read_csv(O/'genuine_sia_oof.csv');prepared=json.loads((O/'prepared.json').read_text());fitrows=json.loads((O/'oof_completion.json').read_text())['results'];plan=json.loads((O/'plan.json').read_text());assert len(fitrows)==len(plan['fits'])==120
    for r in fitrows:
        b=T/r['relative_bundle'];validate_bundle(b);m=r['input_metadata'];assert set(oof[(oof.track==r['track'])&(oof.seed==r['seed'])].asset)==set(m['assets']);assert pd.to_datetime(m['training_decision_times'],utc=True).max()<pd.Timestamp(m['fit_cutoff']);assert pd.to_datetime(m['label_available_at'],utc=True).max()<=pd.Timestamp(m['fit_cutoff']);assert pd.to_datetime(m['label_end'],utc=True).max()<=pd.Timestamp(m['fit_cutoff']);assert m['fit_cutoff']<=min(k[0] for k in m['date_asset_keys']);assert m['predicted_dates']<=13 and m['past_dates']>=26
    class NoTrainingMeter:
        def rows(self):raise AssertionError('Completed bundle tried to touch meter')
    first=fitrows[0];resume=train_fit(None,first['input_metadata'],first['seed'],NoTrainingMeter());assert resume==first
    meterpath=T/'ledger/v33_budget_free_v3_sia_oof_v34_study.sqlite'
    with sqlite3.connect(meterpath.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;meter=[dict(r) for r in db.execute('SELECT * FROM observations ORDER BY rowid')]
    assert sum(r['state']=='SUCCEEDED' for r in meter)==120 and all(r['state'] in ['SUCCEEDED','INTERRUPTED','FAILED'] for r in meter)
    # Validate OOF targets/quantiles and parameters remain immutable after evaluation.
    cal=json.loads((O/'calibration_frozen.json').read_text());assert cal['parameter_id']==digest({k:v for k,v in cal.items() if k!='parameter_id'});assert cal['OOF_hash']==file_hash(O/'genuine_sia_oof.csv')
    nested=cal['parameters']['Nested-B'];assert all(p['supported']==[False,False,True,False,False] or p['supported']==[False]*5 for p in nested.values())
    probe();preservation=preserve();old=json.loads((R/'reports/model_risk_v3/final_handoff.json').read_text());
    for p,h in old['files'].items():assert file_hash(R/p)==h
    p=subprocess.run([sys.executable,'-m','pytest','calibration_v34/test_contract.py','calibration_v34/serialization_r1/test_adapter.py','p1_pit_readiness_v34/test_gates.py','p1_pit_readiness_v34/test_bundle_names.py','tests/v33/test_budget_free_policy.py','-q','-p','no:cacheprovider'],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=':'.join(str(x) for x in [R,R/'src',R/'scripts',R/'studies_v3'])));print(p.stdout+p.stderr,flush=True);assert p.returncode==0
    audit_hash=digest({str(p.relative_to(R)):file_hash(p) for p in sorted((R/'p1_pit_readiness_v34').glob('*.py'))});receipt={'passed':True,'created_at':now(),'configuration_id':config['configuration_id'],'independent_audit_hash':audit_hash,'v34_code_hash':source_hash(),'serialization_revision':json.loads((O/'serialization_repair_v1r1.json').read_text()),'frozen_v2_hash':code_hash(R),'checks':checks,'OOF_rows':len(oof),'outer_rows':len(f[f.model=='SIA_calibrated_v34']),'successful_SIA_OOF_fits':120,'study_observation_seconds':sum(r['elapsed_seconds'] for r in meter),'study_attempts':len(meter),'interrupted_attempts':sum(r['state']=='INTERRUPTED' for r in meter),'failed_attempts':sum(r['state']=='FAILED' for r in meter),'completed_bundle_resume_reused_no_training':True,'tests':{'exit_code':p.returncode,'output':p.stdout+p.stderr},'preservation':preservation,'posthoc_files_preserved':len(old['files']),'reserved_access':False,'final_access':False,'research_qualification':'EXPLORATORY_P0_TIER_B_ONLY'};atomic_json(O/'final_independent_risk_validation.json',receipt)
    status={'state':'COMPLETED_EXECUTABLE_V34_BRANCHES','OOF':'COMPLETED','calibration':'COMPLETED_EXPLORATORY','validation':'PASS','strict_PIT':'BLOCKED_TIER_A','P1':'BLOCKED_ACTION_CLOCK_CASH','future':'WAITING_FOR_GENUINE_FREEZE_AND_FUTURE_OBSERVATION','reserved_final':'SEALED','implementation_complete':False,'reserved_evaluation_complete':False,'strict_final_claims_permitted':False,'fit_count':120,'additional_integration_fits':2,'measured_study_seconds':receipt['study_observation_seconds'],'frozen_v2_hash':code_hash(R),'configuration_id':config['configuration_id'],'final_receipt':'reports/calibration_v34/final_independent_risk_validation.json'};atomic_json(O/'STATUS.json',status)
    # Append-only final index has its own identity; original frozen handoff remains intact.
    files={str(p.relative_to(R)):file_hash(p) for directory in [O,R/'reports/p1_pit_v34'] for p in sorted(directory.iterdir()) if p.is_file() and p.name not in ['final_index.json','status.json']};dest=T/'artifacts/sia_oof_v34/final_index'/digest(files);from p1_pit_readiness_v34.bundle_names import flatten; flat,mapping=flatten({k:(R/k).read_bytes() for k in files}); commit_bundle(dest,flat|{'file_map.json':mapping},{'reserved_access':False,'configuration_id':config['configuration_id']});atomic_json(O/'final_index.json',{'files':files,'relative_bundle':str(dest.relative_to(T)),'independent_audit_hash':audit_hash,'frozen_v2_hash':code_hash(R)});print(json.dumps(status),flush=True)
if __name__=='__main__':main()