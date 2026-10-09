"""Use measured pilot charges only. No predictive metric enters admission."""
from signalforge.runtime import paths,atomic_json,file_hash,code_hash,now
from signalforge.successor import load_frozen_plan,validate_pilot,ledger_snapshot,measured_admission,assert_parent_unchanged,require_successor_access
from signalforge.successor_runtime import progress

repo,runtime=paths();require_successor_access(repo);plan,pointer=load_frozen_plan(repo,runtime)
progress(repo,runtime,'MEASURED BILL ADMISSION',status='OUTCOME_BLIND_COST_ADMISSION')
bill=ledger_snapshot(runtime/'ledger/successor_gpu_pilot_compute.sqlite',3600)
charges={r['id']:r for r in bill['charges']};costs={};pilot_hashes={};parents=[]
for track in plan['tracks']:
    receipt=validate_pilot(repo,runtime,track);parents.append(receipt['parent_ledger'])
    assert_parent_unchanged(runtime,receipt['parent_ledger'])
    pilot_hashes[track]=file_hash(repo/'reports'/('successor_gpu_pilot_'+track.replace('-','_')+'.json'))
    costs[track]={}
    for fit in receipt['results']:
        charge=charges.get(fit['model_id'])
        if not charge or charge['state']!='SUCCEEDED':raise PermissionError('Missing successful measured fit charge')
        costs[track][fit['family']]=charge['seconds']
full=ledger_snapshot(runtime/'ledger/successor_gpu_full_compute.sqlite',21600)
receipt={**measured_admission(plan,costs,max(0,21600-full['charged_seconds'])),
         'created_at':now(),'execution_plan_id':pointer['plan_id'],'pilot_receipt_hashes':pilot_hashes,
         'measured_gpu_seconds_by_track_family':costs,'pilot_ledger':{k:v for k,v in bill.items() if k!='charges'},
         'full_study_ledger_path':str(runtime/plan['full_ledger']),'parent_ledger_unchanged':True,
         'parent_ledger':parents[0],'source_tree_hash':code_hash(repo),'config_hashes':plan['config_hashes']}
atomic_json(repo/'reports/successor_gpu_bill_admission.json',receipt)
print(__import__('json').dumps(receipt,indent=2))
