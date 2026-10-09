import copy,unittest
from signalforge.runtime import digest
from signalforge.v33_minimal_execution import CAPS,LEDGERS,GRID,PROTOCOL,checked,authorize,admit
class ExecutionV2Tests(unittest.TestCase):
    def setUp(self):
        self.p={'protocol_id':PROTOCOL,'reserved_access':False,'caps':CAPS,'ledgers':LEDGERS,'parent_ledger_reuse':False,'execution_source_hash':'source','fits':[dict(zip(['track','year','information','family','seed'],r),settings={'width':16,'lr':.001,'epochs':100},fold_id='f',source_id='s',normalizer_id='n',sia_manifest_id='t') for r in GRID]};self.p['plan_id']=digest(self.p)
        self.a={'protocol_id':PROTOCOL,'plan_id':self.p['plan_id'],'caps':CAPS,'ledgers':LEDGERS,'user_execution_waiver':True,'reserved_access':False};self.a['authorization_id']=digest(self.a)
        self.q={'source_tree_hash':'source','plan_id':self.p['plan_id'],'passed':True,'cuda_sia_fit_predict_reload':True}
        self.b={'source_tree_hash':'source','plan_id':self.p['plan_id'],'ledger_path':LEDGERS['sia'],'fit_count':6,'outcome_blind':True,'safety_margin':1.25,'pilot_receipt_sha256':'pilot','measured_seconds_by_fold':{'Main-A:2020':1,'Nested-B:2022':1},'conservative_projected_seconds':7.5};self.b['bill_id']=digest(self.b)
    def test_exact_admission(self):self.assertEqual(admit(self.p,self.a,self.q,self.b,'source')['state'],'ADMITTED_SIX_FIT_EXECUTION')
    def test_grid_change(self):
        self.p['fits'].pop();self.p['plan_id']=digest({k:v for k,v in self.p.items() if k!='plan_id'})
        with self.assertRaises(PermissionError):checked(self.p)
    def test_caps_change(self):
        self.p['caps']={'pilot':301,'sia':1800,'bar':1800};self.p['plan_id']=digest({k:v for k,v in self.p.items() if k!='plan_id'})
        with self.assertRaises(PermissionError):checked(self.p)
    def test_source_auth(self):
        with self.assertRaises(PermissionError):authorize(self.p,self.a,'changed')
    def test_qualification_absent(self):self.assertEqual(admit(self.p,self.a,{},self.b,'source')['state'],'BLOCKED')
    def test_bill_budget(self):
        self.b['conservative_projected_seconds']=1801;self.b['bill_id']=digest({k:v for k,v in self.b.items() if k!='bill_id'});self.assertEqual(admit(self.p,self.a,self.q,self.b,'source')['state'],'BLOCKED')
    def test_bill_corruption(self):
        self.b['fit_count']=36;self.assertEqual(admit(self.p,self.a,self.q,self.b,'source')['state'],'BLOCKED')
    def test_completed_event_stage_collision_regression(self):
        from pathlib import Path
        source=(Path(__file__).resolve().parents[2]/'scripts/run_v33_minimal_v2.py').read_text(encoding='utf-8-sig')
        self.assertIn("stage_kind=stage,**{k:v for k,v in result.items() if k not in {'source_tree_hash','stage'}}",source)
        def emit(stage,**fields):return {'stage':stage,**fields}
        result={'stage':'integration','source_tree_hash':'source','fit_id':'saved'}
        event=emit('CUDA_FIT_SUCCEEDED',stage_kind='integration',**{k:v for k,v in result.items() if k not in {'source_tree_hash','stage'}})
        self.assertEqual(event['stage'],'CUDA_FIT_SUCCEEDED');self.assertEqual(event['stage_kind'],'integration')
