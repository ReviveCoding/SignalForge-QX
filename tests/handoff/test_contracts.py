from pathlib import Path
import copy,json,sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from handoff_core import stable_hash,stage_key,validate_archive_member,topological_order,convex_fusion,final_allowed,check_common_comparison
from validate_handoff import validate

def load(n):return json.loads((ROOT/n).read_text())
class CoreTests(unittest.TestCase):
 def test_validator(self):self.assertEqual(validate(ROOT)['handoff_contracts'],'PASS')
 def test_hash_order(self):self.assertEqual(stable_hash({'a':1,'b':2}),stable_hash({'b':2,'a':1}))
 def test_hash_change(self):self.assertNotEqual(stable_hash({'a':1}),stable_hash({'a':2}))
 def test_hash_nan(self):
  with self.assertRaises(ValueError):stable_hash({'x':float('nan')})
 def test_stage_unknown(self):
  with self.assertRaises(ValueError):stage_key('foo',{}, {},'x')
 def test_fit_downstream(self):
  with self.assertRaises(ValueError):stage_key('fit',{}, {'calibration_hash':'x'},'x')
 def test_stage_isolation(self):self.assertNotEqual(stage_key('fit',{}, {},'x'),stage_key('report',{}, {},'x'))
 def test_fit_env_change(self):self.assertNotEqual(stage_key('fit',{}, {},'x','a'),stage_key('fit',{}, {},'x','b'))
 def test_archive_normal(self):self.assertEqual(validate_archive_member('SignalForge-QX/docs/a.md')[0],'SignalForge-QX')
 def test_archive_escape(self):
  with self.assertRaises(ValueError):validate_archive_member('SignalForge-QX/../outside')
 def test_archive_absolute(self):
  with self.assertRaises(ValueError):validate_archive_member('/SignalForge-QX/a')
 def test_archive_windows(self):
  with self.assertRaises(ValueError):validate_archive_member('C:\\outside')
 def test_archive_wrongroot(self):
  with self.assertRaises(ValueError):validate_archive_member('Other/a')
 def test_archive_ads(self):
  with self.assertRaises(ValueError):validate_archive_member('SignalForge-QX/a:evil')
 def test_dag(self):self.assertEqual(len(topological_order(load('configs/phases.json')['phases'])),16)
 def test_dag_cycle(self):
  with self.assertRaises(ValueError):topological_order([{'id':'a','depends_on':['b']},{'id':'b','depends_on':['a']}])
 def test_dag_missing(self):
  with self.assertRaises(ValueError):topological_order([{'id':'a','depends_on':['b']}])
 def test_dag_duplicate(self):
  with self.assertRaises(ValueError):topological_order([{'id':'a','depends_on':[]},{'id':'a','depends_on':[]}])
 def test_fallback(self):self.assertEqual(convex_fusion([[1,2,3],[9,10,11]],[1,0]),[1,2,3])
 def test_ordered_fusion(self):
  out=convex_fusion([[-4,-1,0,2,4],[-2,0,1,3,9]],[.4,.6]);self.assertEqual(out,sorted(out))
 def test_simplex(self):
  with self.assertRaises(ValueError):convex_fusion([[1],[2]],[-1,2])
 def test_simplex_sum(self):
  with self.assertRaises(ValueError):convex_fusion([[1],[2]],[.4,.4])
 def test_invalid_expert(self):
  with self.assertRaises(ValueError):convex_fusion([[3,1]],[1])
 def test_empty_fusion(self):
  with self.assertRaises(ValueError):convex_fusion([],[])
 def test_final_denied_empty(self):self.assertFalse(final_allowed({},{}))
 def test_action_approval_not_final(self):self.assertFalse(final_allowed({'approvals_reviewer':'auto_review'},{'status':'READY_FOR_FINAL'}))
 def test_final_requires_receipt(self):self.assertFalse(final_allowed({'authorized':True,'study_id':'sgqx-v3','scope':'one_registered_frozen_batch_after_all_track_gates'},{}))
 def test_final_qualified(self):self.assertTrue(final_allowed({'authorized':True,'study_id':'sgqx-v3','scope':'one_registered_frozen_batch_after_all_track_gates'},{'study_id':'sgqx-v3','status':'READY_FOR_FINAL','all_required_predictive_gates_passed':True,'protocol_hash':'a'}))
 def test_common_grid(self):
  d={k:'x' for k in ['track','price_mode','tier','decision_asset_grid','horizon','normalizer_id']};self.assertTrue(check_common_comparison(d,d))
 def test_common_normalizer_reject(self):
  a={k:'x' for k in ['track','price_mode','tier','decision_asset_grid','horizon','normalizer_id']};b={**a,'normalizer_id':'y'}
  with self.assertRaises(ValueError):check_common_comparison(a,b)
 def test_nested_split(self):self.assertEqual(load('configs/study.json')['splits']['track_folds']['Nested-B']['outer_years'],[2021,2022])
 def test_incremental_report(self):
  p={x['id']:x for x in load('configs/phases.json')['phases']};self.assertEqual(p['P14']['depends_on'],[])
 def test_desk_gpu_independent(self):
  p={x['id']:x for x in load('configs/phases.json')['phases']};self.assertEqual(p['P01']['depends_on'],[])
 def test_predictive_economic_gates(self):self.assertNotIn('price_P1',load('configs/phases.json')['branch_gates']['predictive_final'])
 def test_forward_not_report(self):
  p={x['id']:x for x in load('configs/phases.json')['phases']};self.assertNotIn('P14',p['P15']['depends_on'])
 def test_enough_future_cases(self):self.assertEqual(len(load('contracts/acceptance_tests.json')['cases']),160)
 def test_cases_not_fake_pass(self):self.assertTrue(all(c['test_result']=='NOT_RUN' for c in load('contracts/acceptance_tests.json')['cases']))
 def test_hardware_not_fake(self):
  status=load('IMPLEMENTATION_STATUS.json')
  if status['single_gpu_qualified_on_user_device']:
   audit=load('reports/environment_audit.json')
   self.assertTrue(audit['smoke_hash_matches']);self.assertTrue(audit['smoke']['qualified'])
   self.assertFalse(audit['smoke']['financial_research_result'])
  else:self.assertFalse(status['single_gpu_qualified_on_user_device'])
 def test_codex_empty_prompt(self):self.assertEqual(load('configs/windows_launch.json')['codex_mode'],'interactive_no_initial_message')
 def test_safe_launch(self):
  s=(ROOT/'scripts/Start-InteractiveCodex.ps1').read_text();self.assertIn('approvals_reviewer="auto_review"',s);self.assertIn("'on-request'",s);self.assertIn("'workspace-write'",s);self.assertNotIn('--dangerously-bypass',s)
 def test_no_shell_eval(self):
  s=(ROOT/'scripts/dispatch_wsl.sh').read_text();self.assertNotIn('eval "$',s);self.assertIn('"$@"',s)
 def test_cuda_only_probe(self):
  s=(ROOT/'scripts/gpu_smoke.py').read_text();self.assertIn("'device_type':'cuda'",s);self.assertIn("device.startswith('cuda')",s);self.assertIn('torch.cuda.is_available()',s)
 def test_private_ignored(self):
  s=(ROOT/'.gitignore').read_text();self.assertIn('.local/',s);self.assertIn('.tools/',s);self.assertIn('.env',s)
 def test_stack_candidate_label(self):self.assertIn('not_validated',load('configs/stack_candidate.json')['status'])
 def test_calibration_lowN(self):self.assertGreater(load('configs/statistical_contract.json')['calibration']['minimum_distinct_dates']['0.05'],26)
 def test_cash_excess(self):self.assertIn('as_of_cash_return',load('configs/statistical_contract.json')['cash_objective'])
 def test_ext4_ledger(self):self.assertEqual(load('configs/local_rtx4090_laptop.json')['paths']['ledger_filesystem'],'WSL ext4 only')
 def test_no_driver_installer(self):
  s=(ROOT/'scripts/bootstrap_wsl.sh').read_text();self.assertNotIn('sudo ',s);self.assertNotIn('apt-get install',s);self.assertIn('USE_CUDA=ON',s)
 def test_empty_quantile_vector(self):
  with self.assertRaises(ValueError):convex_fusion([[]],[1])
 def test_environment_receipt_no_fake_model(self):
  d=load('contracts/run.schema.json');self.assertNotIn('model_id',d['required']);self.assertIn('allOf',d)
 def test_v3_schema(self):self.assertEqual(load('contracts/run.schema.json')['$id'],'urn:signalforge:run:v3')
if __name__=='__main__':unittest.main()
