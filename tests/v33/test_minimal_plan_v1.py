"""Bounded structural checks only: no execution authorization or training."""
import json
import unittest
from pathlib import Path
from signalforge.runtime import digest,file_hash,code_hash
from signalforge.v33_admission import checked_plan

ROOT=Path(__file__).resolve().parents[2]
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))

class MinimalPlanV1Tests(unittest.TestCase):
    def setUp(self):
        self.plan=load('configs/v33_minimal_plan_v1.json')
        self.contract=load('contracts/v33_minimal_admission_v1.json')
        self.parent=load('reports/v33_execution_plan.json')
    def test_minimal_identity_hashes(self):
        p=self.plan;c=self.contract
        self.assertEqual(digest({k:v for k,v in p.items() if k!='plan_id'}),p['plan_id'])
        self.assertEqual(digest({k:v for k,v in c.items() if k!='contract_id'}),c['contract_id'])
        self.assertEqual(p['admission_contract_id'],c['contract_id'])
        self.assertEqual(p['source_tree_hash'],code_hash(ROOT))
    def test_minimal_exact_six_combinations(self):
        expected={(t,y,i,'rgmf_gru',s) for t,y,i in [('Main-A',2020,'I3'),('Nested-B',2022,'I4')] for s in [11,37,71]}
        actual={(r['track'],r['year'],r['information'],r['family'],r['seed']) for r in self.plan['fits']}
        self.assertEqual(actual,expected);self.assertEqual(len(self.plan['fits']),6)
        self.assertEqual(self.plan['planned_fits'],6)
        for r in self.plan['fits']:self.assertEqual(r['settings'],{'width':16,'lr':0.001,'epochs':100})
    def test_minimal_exact_two_folds(self):
        exact={'Main-A:2020':'20cd465aa6e731ba2b2e03dce88c5d75f0e09bf63979c53452cf741e0c46b82a','Nested-B:2022':'000b8d624792b1c4fb9a690e63754d8947d1aa0fddc62615aebbc98fccc8ef45'}
        self.assertEqual(set(self.plan['folds']),set(exact))
        for key,foldid in exact.items():
            f=self.parent['folds'][key];small=self.plan['folds'][key]
            self.assertEqual(digest({k:v for k,v in f.items() if k!='fold_id'}),foldid)
            self.assertEqual(small['fold_id'],foldid)
            self.assertGreaterEqual(f['train_distinct_dates'],f['minimum_training_dates'])
            for k in ['source_id','normalizer_id','raw_context_hash']:self.assertEqual(small[k],f[k])
            self.assertEqual(small['sia_manifest_id'],f['sia_manifest']['manifest_id'])
    def test_minimal_legacy_checker_rejects(self):
        checked_plan(self.parent)
        with self.assertRaises(PermissionError):checked_plan(self.plan)
    def test_minimal_fail_closed_no_cost_authorization(self):
        c=self.contract;p=self.plan
        self.assertFalse(c['execution_enabled']);self.assertFalse(p['execution_authorized'])
        self.assertEqual(c['default_state'],'BLOCKED_NOT_AUTHORIZED')
        self.assertIsNone(c['numeric_gpu_ceiling']);self.assertIsNone(c['mini_pilot_numeric_ceiling'])
        self.assertFalse(c['parent_ledger_reuse']);self.assertFalse(c['reserved_access'])
        self.assertNotIn('development_compute.sqlite',c['ledger_path'])
        self.assertNotEqual(c['ledger_path'],c['mini_pilot_ledger_path'])
        for gate in ['DEDICATED_SIX_FIT_CHECKER_IMPLEMENTED_AND_TESTED','EXPLICIT_USER_NUMERIC_CEILING_AND_MINI_PILOT_CEILING','IMMUTABLE_OUTCOME_BLIND_MEASURED_BILL_WITH_SAFETY_MARGIN','CURRENT_SOURCE_BOUND_SIX_FIT_CUDA_EXECUTION_VALIDATION']:
            self.assertIn(gate,c['required_before_any_launch'])
    def test_minimal_baseline_scope_and_deferred_work(self):
        p=self.plan
        self.assertEqual(p['historical_reference_scores']['Main-A'],{'family':'historical','score':0.246973383})
        self.assertEqual(p['historical_reference_scores']['Nested-B'],{'family':'linear_quantile','score':0.215321149})
        self.assertTrue(p['comparison_contract']['all_rows_required'])
        self.assertIsNone(p['comparison_contract']['new_performance_claim'])
        self.assertFalse(p['comparison_contract']['new_predictions_exist'])
        self.assertEqual(p['new_gpu_fits'],0);self.assertFalse(p['final_qualified'])
        for item in ['ABC','full HPO','source certification','Nested-B 2021 underpowered branch','2392-fit forward v3.2','remaining 36-fit programme']:self.assertIn(item,p['deferred'])
    def test_minimal_all_previous_files_preserved(self):
        baseline=load('reports/v33_minimal_preservation_baseline_v1.json')
        for name,h in baseline['files'].items():self.assertEqual(file_hash(ROOT/name),h,name)
        original=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX')
        isolation=load('reports/v33_isolation_receipt.json')
        for name,h in isolation['copied_files'].items():self.assertEqual(file_hash(original/name.replace('\\','/')),h,name)
        self.assertEqual(code_hash(original),baseline['original_source_hash'])

if __name__=='__main__':unittest.main()
