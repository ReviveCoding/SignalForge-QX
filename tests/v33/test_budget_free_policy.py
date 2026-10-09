"""Targeted CPU-only policy/meter tests; no CUDA fitting or research evidence."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
from signalforge.runtime import digest
from policy_v3 import v33_budget_free as v

class Clock:
    value=0.
    def __call__(self):return self.value

class MeterTests(unittest.TestCase):
    def setUp(self):
        v.LEDGER_ROOT.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=v.LEDGER_ROOT);self.path=Path(self.tmp.name)/'v33_budget_free_v3_test.sqlite';self.clock=Clock();self.m=v.ObservationMeter(self.path,clock=self.clock)
    def tearDown(self):self.m.close();self.tmp.cleanup()
    def test_no_ceiling_and_persistence(self):
        with self.m.attempt('long') as a:
            self.clock.value=6000.;a.observe(peak_vram_bytes=256,outcome={'loss':.4});self.clock.value=24000.
        self.m.close();self.m=v.ObservationMeter(self.path,clock=self.clock);row=self.m.rows()[0]
        self.assertEqual(row['elapsed_seconds'],24000);self.assertEqual(row['state'],'SUCCEEDED');self.assertEqual(row['peak_vram_bytes'],256);self.assertEqual(json.loads(row['outcome']),{'loss':.4})
        self.assertEqual(self.m.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(),[('observations',)])
    def test_failure_recorded_without_exception_text(self):
        with self.assertRaises(RuntimeError):
            with self.m.attempt('failed'):self.clock.value=4;raise RuntimeError('sensitive message never persisted')
        row=self.m.rows()[0];self.assertEqual(row['state'],'FAILED');self.assertEqual(row['failure_type'],'RuntimeError');self.assertNotIn('sensitive',str(row))
    def test_cooperative_cancel(self):
        with self.assertRaises(v.CooperativeCancellation):
            with self.m.attempt('cancel') as a:a.cancel()
        self.assertEqual(self.m.rows()[0]['state'],'INTERRUPTED')
    def test_keyboard_interrupted(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.m.attempt('interrupt'):raise KeyboardInterrupt()
        self.assertEqual(self.m.rows()[0]['state'],'INTERRUPTED')
    def test_duplicate_fail_closed(self):
        with self.m.attempt('same'):pass
        with self.assertRaises(ValueError):
            with self.m.attempt('same'):pass
        self.assertEqual(len(self.m.rows()),1)
    def test_explicit_retry_and_bounded_incident_count(self):
        for name,parent in [('root',None),('r1','root'),('r2','r1')]:
            with self.assertRaises(RuntimeError):
                with self.m.attempt(name,retry_of=parent):raise RuntimeError()
        with self.assertRaises(PermissionError):
            with self.m.attempt('r3',retry_of='r2'):pass
        self.assertEqual(len(self.m.rows()),3)
    def test_success_cannot_retry(self):
        with self.m.attempt('ok'):pass
        with self.assertRaises(PermissionError):
            with self.m.attempt('unneeded',retry_of='ok'):pass
    def test_dead_owner_recovered_last_heartbeat_only(self):
        with self.m.attempt('dead') as a:self.clock.value=7;a.observe()
        with self.m.db:self.m.db.execute("UPDATE observations SET state='RUNNING',pid=99999999,elapsed_complete=0 WHERE identity='dead'")
        self.m.close();self.m=v.ObservationMeter(self.path,clock=self.clock);row=self.m.rows()[0]
        self.assertEqual(row['state'],'INTERRUPTED');self.assertEqual(row['elapsed_seconds'],7);self.assertEqual(row['elapsed_complete'],0)
    def test_alive_owner_not_reclassified(self):
        with self.m.attempt('active'):
            other=v.ObservationMeter(self.path,clock=self.clock)
            try:self.assertEqual(other.rows()[0]['state'],'RUNNING')
            finally:other.close()
    def test_old_ledgers_denied(self):
        for path in [v.LEDGER_ROOT/'development_compute.sqlite',v.LEDGER_ROOT/'v33_minimal_v2_sia_compute.sqlite',Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1/ledger/v33_budget_free_v3_wrong.sqlite')]:
            with self.assertRaises(PermissionError):v.ObservationMeter(path)
    def test_invalid_outcome_and_vram_record_failed(self):
        with self.assertRaises(ValueError):
            with self.m.attempt('bad') as a:a.observe(outcome={'loss':float('nan')})
        self.assertEqual(self.m.rows()[0]['state'],'FAILED')
        with self.assertRaises(ValueError):
            with self.m.attempt('bad-vram') as a:a.observe(peak_vram_bytes=-1)

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        keys=[['2022-01-07T23:00:00+00:00','SPY'],['2022-01-07T23:00:00+00:00','QQQ']]
        fold={'source_id':'source','schema_id':'schema','normalizer_id':'scale','target_id':'target','date_asset_keys_hash':digest(keys),'date_asset_keys':keys,'assets':['SPY','QQQ'],'fit_cutoff':'2021-12-31T23:00:00+00:00','training_decision_times':['2021-12-01T23:00:00+00:00'],'label_end':['2021-12-10T23:00:00+00:00'],'label_available_at':['2021-12-11T23:00:00+00:00']}
        fits=[{'fold_id':'fold','seed':s,**{k:fold[k] for k in ['source_id','schema_id','normalizer_id','target_id','date_asset_keys_hash']}} for s in [11,37,71]]
        self.q={'policy_source_hash':v.policy_source_hash(),'source_tree_hash':'current','passed':True,'cuda_inputs_validated':True,'plan_contract_id':'registered','folds':{'fold':fold},'validated_fits':fits}
        self.p={'policy_source_hash':v.policy_source_hash(),'policy_id':v.POLICY_ID,'policy_hash':digest(v.policy()),'source_tree_hash':'current','local_only':True,'reserved_access':False,'final_access':False,'plan_contract_id':'registered','fits':fits};self.bind()
    def bind(self):
        self.q['qualification_id']=digest({k:x for k,x in self.q.items() if k!='qualification_id'});self.p['qualification_id']=self.q['qualification_id'];self.p['plan_id']=digest({k:x for k,x in self.p.items() if k!='plan_id'})
    def run_admission(self):return v.admission(self.p,self.q,experiment_requested=True,current_source_hash='current')
    def test_no_bill_ceiling_or_manual_approval(self):
        self.assertEqual(self.run_admission()['state'],'READY_FOR_DEVICE_LEASE');self.assertFalse(self.run_admission()['training_started']);self.assertFalse(self.run_admission()['manual_cost_approval_required'])
    def test_no_autostart(self):
        with self.assertRaises(PermissionError):v.admission(self.p,self.q,current_source_hash='current')
    def test_plan_hash_corruption(self):
        self.p['fits']=[]
        with self.assertRaises(PermissionError):self.run_admission()
    def test_source_binding(self):
        with self.assertRaises(PermissionError):v.admission(self.p,self.q,experiment_requested=True,current_source_hash='different')
    def test_qualification_not_validated(self):
        self.q['cuda_inputs_validated']=False;self.bind()
        with self.assertRaises(PermissionError):self.run_admission()
    def test_reserved_and_final_fail_closed(self):
        for key in ['reserved_access','final_access']:
            self.p[key]=True;self.bind()
            with self.assertRaises(PermissionError):self.run_admission()
            self.p[key]=False
    def test_wrong_fixed_seed_grid(self):
        self.p['fits']=[dict(f,seed=99) for f in self.p['fits']];self.bind()
        with self.assertRaises(PermissionError):self.run_admission()
    def test_immature_label(self):
        self.q['folds']['fold']['label_available_at']=['2022-01-01T00:00:00+00:00'];self.bind()
        with self.assertRaises(PermissionError):self.run_admission()
    def test_reserved_date_grid(self):
        f=self.q['folds']['fold'];f['date_asset_keys']=[['2024-01-01T00:00:00+00:00','SPY']];f['date_asset_keys_hash']=digest(f['date_asset_keys']);self.bind()
        with self.assertRaises(PermissionError):self.run_admission()
    def test_asset_identity(self):
        self.q['folds']['fold']['assets']=['SPY'];self.bind()
        with self.assertRaises(PermissionError):self.run_admission()
    def test_receipt_corruption(self):
        self.q['folds']['fold']['target_id']='changed'
        with self.assertRaises(PermissionError):self.run_admission()

class DeviceTests(unittest.TestCase):
    def test_desktop_utilization_not_gate(self):self.assertTrue(v.check_device(name='RTX 4090 Laptop GPU',free_vram_bytes=8*1024**3,required_vram_bytes=2*1024**3,compute_pids=[],desktop_utilization=99))
    def test_compute_owner_blocks(self):
        with self.assertRaises(v.DeviceSafetyError):v.check_device(name='RTX 4090 Laptop GPU',free_vram_bytes=8*1024**3,required_vram_bytes=2*1024**3,compute_pids=[999],own_pid=1)
    def test_vram_headroom(self):
        with self.assertRaises(v.DeviceSafetyError):v.check_device(name='RTX 4090 Laptop GPU',free_vram_bytes=2*1024**3,required_vram_bytes=2*1024**3,compute_pids=[])
    def test_wrong_gpu(self):
        with self.assertRaises(v.DeviceSafetyError):v.check_device(name='different GPU',free_vram_bytes=8*1024**3,required_vram_bytes=1,compute_pids=[])
    def test_lease_and_oom_no_fallback(self):
        with tempfile.TemporaryDirectory(dir=v.LEDGER_ROOT) as tmp:
            lock=Path(tmp)/'shared.lock';lock.touch()
            with patch.object(v,'SHARED_LOCK',lock),patch.object(v,'ensure_gpu_owner') as owner,patch.object(v,'gpu_lease',return_value=nullcontext()),patch('subprocess.run',return_value=SimpleNamespace(stdout='RTX 4090 Laptop GPU, 10000, 99')):
                with self.assertRaises(v.DeviceSafetyError):
                    with v.device_lease(1024**3):raise RuntimeError('CUDA out of memory')
                owner.assert_called_once()
    def test_exclusive_lease_conflict(self):
        import fcntl
        with tempfile.TemporaryDirectory(dir=v.LEDGER_ROOT) as tmp:
            lock=Path(tmp)/'shared.lock';lock.touch()
            with lock.open('rb') as held:
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(v,'SHARED_LOCK',lock):
                    with self.assertRaises(BlockingIOError):
                        with v.device_lease(1):pass

class FrozenSourceHashTests(unittest.TestCase):
    def test_actual_frozen_v2_hash(self):
        self.assertEqual(v.code_hash(v.REPO),'471e70eec0fd912ae1ae652efb9f494f3cc28b371c4f5c0e4d2e7f21abb5e870')
        self.assertEqual(Path(v.__file__).resolve().parent,v.REPO/'policy_v3')
        self.assertFalse((v.REPO/'src/signalforge/v33_budget_free.py').exists())
    def test_external_policy_hash_is_separately_bound(self):
        fixture=AdmissionTests();fixture.setUp();fixture.p['policy_source_hash']='stale';fixture.bind()
        with self.assertRaises(PermissionError):fixture.run_admission()
        fixture.setUp();fixture.q['policy_source_hash']='stale';fixture.bind()
        with self.assertRaises(PermissionError):fixture.run_admission()
