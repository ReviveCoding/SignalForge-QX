"""Read-only independent readiness verification and targeted tests, new receipts only."""
import json,sys,subprocess
from pathlib import Path
import pandas as pd
from signalforge.runtime import file_hash,digest,atomic_json,validate_bundle,code_hash
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');O=R/'reports/p1_pit_v34';T=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering')
def main():
    p=subprocess.run([sys.executable,'-m','pytest','calibration_v34/test_contract.py','p1_pit_readiness_v34/test_gates.py','tests/v33/test_budget_free_policy.py','-q','-p','no:cacheprovider'],capture_output=True,text=True);print(p.stdout+p.stderr,flush=True);atomic_json(O/'tests.json',{'exit_code':p.returncode,'output':p.stdout+p.stderr,'scope':'v34 time/calibration and source gates plus unchanged v3 budget-free policy','frozen_v2_hash':code_hash(R)});assert p.returncode==0
    d=json.loads((O/'readiness.json').read_text());events=pd.read_csv(O/'per_version_clock_audit.csv');assert len(events)==67448==sum(s['rows'] for s in d['sources']);assert events.certified_first_public.sum()==0
    gaps=pd.read_csv(O/'p1_per_session_gaps.csv');assert len(gaps)==d['P1']['per_session_gap_rows'];assert set(gaps.asset)=={'SPY','QQQ','IEF','TLT','GLD','SLV','USO','UNG'};assert (pd.to_datetime(gaps.date)<pd.Timestamp('2024-01-01')).all();assert len(pd.read_csv(O/'p1_asset_year_gaps.csv'))==112
    for s in d['official_public_retrievals']:
        if 'relative_bundle' in s:
            b=T/s['relative_bundle'];validate_bundle(b);assert file_hash(b/'document.pdf')==s['sha256']
    old=json.loads((R/'reports/model_risk_v3/final_handoff.json').read_text());
    for name,h in old['files'].items():assert file_hash(R/name)==h,name
    if 'relative_bundle' in old:validate_bundle(T/old['relative_bundle'])
    from diagnostics_v3.audit import preserve
    preservation=preserve();h=digest({str(p.relative_to(R)):file_hash(p) for p in sorted((R/'p1_pit_readiness_v34').glob('*')) if p.suffix=='.py'});atomic_json(O/'independent_validation.json',{'passed':True,'readiness_code_hash':h,'frozen_v2_hash':code_hash(R),'original_preservation':preservation,'posthoc_files_preserved':len(old['files']),'version_rows':len(events),'P1_gap_rows':len(gaps),'all_scientific_gates_remain_blocked':True,'GPU_training':0,'reserved_access':False})
    print(json.dumps({'source_readiness_validation':'PASS','version_rows':len(events),'P1_gap_rows':len(gaps),'frozen_v2_hash':code_hash(R)}),flush=True)
if __name__=='__main__':main()