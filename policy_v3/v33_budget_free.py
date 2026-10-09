"""Future local-development policy: persistent observations, never a time allowance.
No training is launched here. Existing frozen runners must not be retrofitted.
"""
import json,math,os,sqlite3,time
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from signalforge.runtime import digest,code_hash,file_hash,ensure_gpu_owner,gpu_lease

POLICY_ID='sgqx-v33-local-budget-free-v3'
REPO=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev')
RUNTIME=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering')
LEDGER_ROOT=RUNTIME/'ledger'
METER_PATH=LEDGER_ROOT/'v33_budget_free_v3_observations.sqlite'
SHARED_LOCK=Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1/locks/gpu0.lock')

def policy():
    config=json.loads((REPO/'configs/v33_budget_free_policy_v3.json').read_text(encoding='utf-8-sig'))
    if config['policy_id']!=POLICY_ID or any(config[k] is not None for k in ['gpu_seconds_ceiling','pilot_seconds_ceiling','per_fit_timeout_seconds','desktop_utilization_threshold']):raise PermissionError('V3 observation-only policy identity')
    if any(config[k] is not False for k in ['manual_cost_approval_required','measured_bill_required','reserved_access','final_access','auto_start_training']):raise PermissionError('V3 scope/approval policy')
    if config['max_explicit_retries']!=2 or config['vram_headroom_bytes']<1024**3:raise PermissionError('Recovery/device safety policy')
    return config

def policy_source_hash():return file_hash(Path(__file__))

class DeviceSafetyError(RuntimeError):pass
class CooperativeCancellation(Exception):pass

def utcnow():return datetime.now(timezone.utc).isoformat()
def _boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def _process_start(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None

def _clock(value):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise PermissionError('Aware scientific clocks required')
    return dt.astimezone(timezone.utc)

class Attempt:
    def __init__(self,meter,identity,start):self.meter=meter;self.identity=identity;self.start=start;self.peak_vram_bytes=0;self.outcome=None
    def observe(self,*,peak_vram_bytes=None,outcome=None):
        if peak_vram_bytes is not None:
            if isinstance(peak_vram_bytes,bool) or not isinstance(peak_vram_bytes,int) or peak_vram_bytes<0:raise ValueError('Nonnegative integer VRAM')
            self.peak_vram_bytes=max(self.peak_vram_bytes,peak_vram_bytes)
        if outcome is not None:json.dumps(outcome,allow_nan=False);self.outcome=outcome
        elapsed=self.meter._elapsed(self.start)
        with self.meter.db:self.meter.db.execute('UPDATE observations SET elapsed_seconds=?,peak_vram_bytes=?,outcome=? WHERE identity=?',(elapsed,self.peak_vram_bytes,json.dumps(self.outcome,allow_nan=False),self.identity))
    def cancel(self):raise CooperativeCancellation('Caller requested cancellation')

class ObservationMeter:
    """Records observed wall seconds while using GPU, with no cap/reservations.
    Heartbeat via Attempt.observe. Dead-owner recovery retains last measured
    elapsed as a lower bound; never invents time for an unobserved crash tail.
    """
    def __init__(self,path=METER_PATH,*,clock=time.monotonic):
        path=Path(path).resolve()
        if not path.is_relative_to(LEDGER_ROOT.resolve()) or not path.name.startswith('v33_budget_free_v3_') or path.suffix!='.sqlite':raise PermissionError('Separate v3 ext4 observation meter required')
        path.parent.mkdir(parents=True,exist_ok=True);self.clock=clock;self.db=sqlite3.connect(path,timeout=5)
        self.db.execute('PRAGMA journal_mode=WAL')
        with self.db:self.db.execute('''CREATE TABLE IF NOT EXISTS observations(identity TEXT PRIMARY KEY,retry_of TEXT,root_identity TEXT NOT NULL,state TEXT NOT NULL,started_at TEXT NOT NULL,ended_at TEXT,elapsed_seconds REAL NOT NULL,elapsed_complete INTEGER NOT NULL,peak_vram_bytes INTEGER NOT NULL,outcome TEXT,failure_type TEXT,metadata TEXT NOT NULL,pid INTEGER NOT NULL,boot_id TEXT NOT NULL,process_start TEXT NOT NULL)''')
        self.recover_interrupted()
    def _elapsed(self,start):
        elapsed=float(self.clock()-start)
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid observation clock')
        return elapsed
    def recover_interrupted(self):
        rows=self.db.execute("SELECT identity,pid,boot_id,process_start FROM observations WHERE state='RUNNING'").fetchall()
        with self.db:
            for identity,pid,boot,start in rows:
                if boot!=_boot() or _process_start(pid)!=start:self.db.execute("UPDATE observations SET state='INTERRUPTED',ended_at=?,failure_type='OWNER_EXIT_OR_REBOOT',elapsed_complete=0 WHERE identity=?",(utcnow(),identity))
    @contextmanager
    def attempt(self,identity,*,metadata=None,retry_of=None):
        if not isinstance(identity,str) or not identity:raise ValueError('Explicit nonempty identity required')
        metadata=json.dumps(metadata or {},sort_keys=True,allow_nan=False);start=self.clock();self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM observations WHERE identity=?',(identity,)).fetchone():raise ValueError('Duplicate attempt identity')
            root=identity
            if retry_of is not None:
                parent=self.db.execute('SELECT root_identity,state FROM observations WHERE identity=?',(retry_of,)).fetchone()
                if parent is None or parent[1] not in {'FAILED','INTERRUPTED'}:raise PermissionError('Retry requires identified unsuccessful terminal attempt')
                root=parent[0];count=self.db.execute('SELECT COUNT(*) FROM observations WHERE root_identity=? AND retry_of IS NOT NULL',(root,)).fetchone()[0]
                if count>=policy()['max_explicit_retries']:raise PermissionError('Explicit retry count exhausted; diagnose incident')
            self.db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(identity,retry_of,root,'RUNNING',utcnow(),None,0.,0,0,None,None,metadata,os.getpid(),_boot(),_process_start(os.getpid())))
            self.db.commit()
        except BaseException:self.db.rollback();raise
        record=Attempt(self,identity,start);state='SUCCEEDED';failure=None
        try:yield record
        except BaseException as error:
            state='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit,CooperativeCancellation)) else 'FAILED';failure=type(error).__name__;raise
        finally:
            elapsed=self._elapsed(start)
            with self.db:self.db.execute('UPDATE observations SET state=?,ended_at=?,elapsed_seconds=?,elapsed_complete=1,peak_vram_bytes=?,outcome=?,failure_type=? WHERE identity=?',(state,utcnow(),elapsed,record.peak_vram_bytes,json.dumps(record.outcome,allow_nan=False),failure,identity))
    def rows(self):
        cursor=self.db.execute('SELECT * FROM observations ORDER BY rowid');names=[c[0] for c in cursor.description];return [dict(zip(names,row)) for row in cursor.fetchall()]
    def close(self):self.db.close()

def admission(plan,qualification,*,experiment_requested=False,current_source_hash=None):
    """Mechanical scientific integrity admission, not final/Tier-A permission.
    Qualification must come from a real current CUDA input validation receipt.
    No cost bill, pilot duration, budget amount or cost-approval argument exists.
    """
    h=code_hash(REPO) if current_source_hash is None else current_source_hash
    if not experiment_requested:raise PermissionError('New experiment not requested; no automatic launch')
    if plan.get('plan_id')!=digest({k:v for k,v in plan.items() if k!='plan_id'}):raise PermissionError('Plan hash')
    if plan.get('policy_id')!=POLICY_ID or plan.get('policy_source_hash')!=policy_source_hash() or plan.get('policy_hash')!=digest(policy()) or plan.get('source_tree_hash')!=h or plan.get('local_only') is not True or plan.get('reserved_access') is not False or plan.get('final_access') is not False:raise PermissionError('Source/policy/scope')
    if qualification.get('qualification_id')!=digest({k:v for k,v in qualification.items() if k!='qualification_id'}):raise PermissionError('CUDA receipt hash')
    if qualification.get('source_tree_hash')!=h or qualification.get('policy_source_hash')!=policy_source_hash() or qualification.get('passed') is not True or qualification.get('cuda_inputs_validated') is not True or qualification.get('plan_contract_id')!=plan.get('plan_contract_id'):raise PermissionError('Current real CUDA inputs required')
    if plan.get('qualification_id')!=qualification['qualification_id']:raise PermissionError('Exact qualification binding')
    fits=plan.get('fits',[]);folds=qualification.get('folds',{});expected=qualification.get('validated_fits',[])
    if not fits or fits!=expected or len({digest(f) for f in fits})!=len(fits):raise PermissionError('Fixed study asset/date/fold/seed identity')
    for fit in fits:
        fold=folds.get(fit['fold_id']);seed=fit.get('seed')
        if fold is None or isinstance(seed,bool) or not isinstance(seed,int):raise PermissionError('Registered fold/seed')
        for key in ['source_id','schema_id','normalizer_id','target_id','date_asset_keys_hash']:
            if not fold.get(key) or fit.get(key)!=fold[key]:raise PermissionError('Source/schema/normalizer/target/grid identity')
        keys=fold.get('date_asset_keys',[])
        if not keys or digest(keys)!=fold['date_asset_keys_hash'] or len({tuple(k) for k in keys})!=len(keys):raise PermissionError('Exact unique date/asset grid')
        assets=set(fold.get('assets',[]))
        if {asset for date,asset in keys}!=assets or not assets:raise PermissionError('Exact assets')
        boundary=datetime(2024,1,1,tzinfo=timezone.utc)
        if any(_clock(date)>=boundary for date,asset in keys):raise PermissionError('Reserved development origin')
        cutoff=_clock(fold['fit_cutoff']);dates=fold.get('training_decision_times',[]);ends=fold.get('label_end',[]);available=fold.get('label_available_at',[])
        if cutoff>=boundary or not dates or len(dates)!=len(ends) or len(dates)!=len(available):raise PermissionError('Complete maturity lineage')
        for date,end,at in zip(dates,ends,available):
            date,end,at=map(_clock,(date,end,at))
            if not date<end<=at<=cutoff or date>=boundary:raise PermissionError('Immature or reserved training label')
    return {'state':'READY_FOR_DEVICE_LEASE','policy_id':POLICY_ID,'plan_id':plan['plan_id'],'training_started':False,'meter_only':True,'manual_cost_approval_required':False,'reserved_access':False,'qualified_for_final':False}

def check_device(*,name,free_vram_bytes,required_vram_bytes,compute_pids,own_pid=None,desktop_utilization=None):
    if '4090 Laptop' not in name:raise DeviceSafetyError('Registered local physical GPU required')
    if any(not isinstance(pid,int) or pid!=(os.getpid() if own_pid is None else own_pid) for pid in compute_pids):raise DeviceSafetyError('Competing or unknown CUDA compute owner')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in [free_vram_bytes,required_vram_bytes]):raise DeviceSafetyError('Valid VRAM estimate required')
    if free_vram_bytes<required_vram_bytes+policy()['vram_headroom_bytes']:raise DeviceSafetyError('VRAM safety headroom')
    return True # Windows desktop utilization is observational, never a hard gate.

@contextmanager
def device_lease(required_vram_bytes):
    import fcntl,subprocess
    # Coordinate with existing original lease read-only; never edits its record.
    with SHARED_LOCK.open('rb') as shared:
        fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            ensure_gpu_owner()
            obs=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip().split(',')
            if len(obs)!=3:raise DeviceSafetyError('Unknown GPU inventory')
            check_device(name=obs[0],free_vram_bytes=int(obs[1])*1024**2,required_vram_bytes=required_vram_bytes,compute_pids=[],desktop_utilization=float(obs[2]))
            with gpu_lease(RUNTIME):yield {'gpu_name':obs[0],'desktop_utilization_observed':float(obs[2])}
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():raise DeviceSafetyError('CUDA_OOM: preserve meter/failure and diagnose; no silent fallback') from error
            raise
        finally:fcntl.flock(shared,fcntl.LOCK_UN)
