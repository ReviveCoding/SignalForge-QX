"""Conservative project inspection register; no reserved source/label reads."""
import json
from signalforge.runtime import paths,atomic_json,file_hash,now
from signalforge.contamination import record_inspection,assert_unused_period

repo,runtime=paths();cards=[]
for name,purpose in [('cftc_development_acquisition.json','2010-2023 annual archive parsing/QA'),
                     ('eia_development_acquisition.json','pre-2024 dated archive parsing/source QA'),
                     ('auxiliary_cpu_postprocessing.json','2023 registered selection/calibration diagnostics')]:
    path=repo/'reports'/name
    if not path.exists():continue
    record={'start':'2010-01-01T00:00Z','end':'2024-01-01T00:00Z','purpose':purpose,
        'evidence_path':str(path),'evidence_sha256':file_hash(path),'scientific_partition':'development_and_registered_2023_postprocessing',
        'interval_scope':'conservative union of already parsed/inspected pre-2024 years; not a claim every date had a qualified target'}
    identity=record_inspection(runtime,record);cards.append({**record,'inspection_id':identity})
assert_unused_period(cards,'2024-01-01T00:00Z','2026-07-01T00:00Z')
admitted=(runtime/'ledger/final_access.json').exists()
result={'state':'RECORDED_REGISTERED_WORKFLOW_INSPECTION_HISTORY','created_at':now(),'records':cards,
    'registered_reserved_interval':['2024-01-01T00:00Z','2026-07-01T00:00Z'],
    'scientific_admission_ledger_present':admitted,'reserved_payload_read_by_this_audit':False,
    'known_workflow_reserved_state':'ADMISSION_RECORDED_REQUIRES_FINAL_EVENT_RECONCILIATION' if admitted else 'NO_REGISTERED_SCIENTIFIC_ADMISSION',
    'prior_external_inspection_history':'UNKNOWN_NOT_CERTIFIED_BY_ABSENCE_OF_PROJECT_LEDGER',
    'unseen_cohort_scientifically_qualified':False,'renaming_consumed_dates_allowed':False}
atomic_json(repo/'reports/contamination_register.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
