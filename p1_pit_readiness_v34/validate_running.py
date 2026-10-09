"""Read-only current successful CUDA bundles/checkpoints; CPU integrity checks only."""
import json,sqlite3
import torch,numpy as np
from pathlib import Path
from signalforge.runtime import validate_bundle,file_hash,atomic_json,now
from calibration_v34.runner import R,T,O,cfg

def main():
    c=cfg();path=T/'ledger/v33_budget_free_v3_sia_oof_v34_study.sqlite'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;rows=[dict(r) for r in db.execute('SELECT * FROM observations ORDER BY rowid')]
    successful=[r for r in rows if r['state']=='SUCCEEDED'];ids=[]
    for row in successful:
        b=T/'artifacts/sia_oof_v34/study'/row['root_identity'];validate_bundle(b);r=json.loads((b/'fit.json').read_text());state=torch.load(b/'model.pt',map_location='cpu',weights_only=False);config=state['config'];assert config['kind']=='rgmf_gru' and config['width']==16 and config['lr']==.001 and config['epochs']==100 and config['seed']==r['seed'];assert r['configuration_id']==c['configuration_id'];assert all(torch.isfinite(v).all() for v in state['model'].values());assert len(state['losses'])==100 and np.isfinite(state['losses']).all();assert r['reload_bitwise_equal'];assert file_hash(r['checkpoint_path'])==r['checkpoint_hash'];ck=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False);assert ck['step']==100 and ck['sample_cursor']==100*len(r['input_metadata']['training_decision_times']);p=np.load(b/'predictions.npz');assert np.isfinite(p['quantiles']).all() and (np.diff(p['quantiles'],axis=1)>=0).all();ids.append(r['identity'])
    evidence={'checked_at':now(),'completed_models_verified':len(ids),'model_ids':ids,'interrupted_attempts_retained':sum(r['state']=='INTERRUPTED' for r in rows),'failed_attempts':sum(r['state']=='FAILED' for r in rows),'state':'PARTIAL_INTEGRITY_PASS' if len(ids)<120 else 'ALL_SAVED_MODEL_INTEGRITY_PASS','additional_GPU_training':0,'CUDA_reload_before_commit_observed_in_each_fit':True,'CPU_inference_used':False};atomic_json(O/'running_bundle_integrity.json',evidence);print(json.dumps({k:v for k,v in evidence.items() if k!='model_ids'}),flush=True)
if __name__=='__main__':main()