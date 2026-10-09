"""Validate the delivered design/setup contracts, not unimplemented financial code."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from handoff_core import topological_order

def validate(root):
    load=lambda f:json.loads((root/f).read_text(encoding='utf-8'))
    study=load('configs/study.json');phases=load('configs/phases.json');exp=load('configs/experiments.json')
    acc=load('contracts/acceptance_tests.json');profile=load('configs/local_rtx4090_laptop.json');launch=load('configs/windows_launch.json')
    assert study['study_id']=='sgqx-v3'
    assert len(phases['phases'])==16
    topological_order(phases['phases'])
    assert len(exp['experiments'])==10
    assert len(acc['cases'])==160
    assert len({c['id'] for c in acc['cases']})==160
    assert launch['approvals_reviewer']=='auto_review' and launch['approval_policy']=='on-request'
    assert launch['sandbox']=='workspace-write' and launch['codex_mode']=='interactive_no_initial_message'
    assert not launch['modify_global_codex_config'] and not launch['disable_sandbox']
    assert profile['cuda']['max_heavy_jobs']==1 and not profile['cuda']['neural_cpu_fallback']
    assert profile['paths']['repo']==r'C:\Users\USERNAME\Downloads\SignalForge-QX'
    assert study['evaluation']['shared_normalizer_per_comparison']
    for fn in ['CODEX_MASTER_PROMPT.md','AGENTS.md','docs/FINAL_FRAMEWORK_V3_KO.md','scripts/Prepare-Workspace.ps1','scripts/Start-InteractiveCodex.ps1','scripts/bootstrap_wsl.sh','scripts/gpu_smoke.py']:
        assert (root/fn).is_file(),fn
    byid={e['id']:e for e in exp['experiments']}
    for x in ['E04','E07','E08','E10']:assert byid[x]['development_only']
    return {'handoff_contracts':'PASS','phases':16,'experiments':10,'future_acceptance_requirements':160,
            'full_research_implemented':False,'user_GPU_qualified_here':False}

def verify_manifest(root):
    m=json.loads((root/'MANIFEST.json').read_text());n=0
    for f in m['files']:
        p=root/f['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Modified file: '+f['path'])
        n+=1
    return n
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-manifest',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    result=validate(root)
    if a.verify_manifest:result['manifest_files_verified']=verify_manifest(root)
    print(json.dumps(result,indent=2))
