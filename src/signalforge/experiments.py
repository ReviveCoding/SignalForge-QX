"""Outcome-blind experiment admission and persistent attempt registry."""
import json
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager
from .runtime import digest


class TrialLedger:
    def __init__(self,path):
        path=Path(path)
        if str(path.resolve()).startswith('/mnt/'):
            raise ValueError('SQLite ledger must reside on ext4')
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, configuration TEXT NOT NULL, state TEXT NOT NULL, metric REAL, failure TEXT)')
        self.db.commit()

    def register(self,configuration):
        if configuration.get('partition') not in {'inner_train','inner_validation'}:
            raise PermissionError('HPO may register only inner temporal partitions')
        if configuration.get('experiment')=='E04' and configuration.get('promotion_eligible') is not False:
            raise ValueError('Invalid leakage controls can never be promoted')
        identity=digest(configuration)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?,NULL,NULL)',
                            (identity,json.dumps(configuration,sort_keys=True),'PLANNED'))
        return identity

    def finish(self,identity,state,metric=None,failure=None):
        if state not in {'SUCCEEDED','FAILED_PERMANENT','FAILED_RETRYABLE','PRUNED','INTERRUPTED','PAUSED_BUDGET'}:
            raise ValueError('Invalid trial completion state')
        with self.db:
            current=self.db.execute('SELECT state FROM attempts WHERE id=?',(identity,)).fetchone()
            if not current or current[0] not in {'PLANNED','RUNNING'}:
                raise ValueError('Attempt missing or already finalized')
            self.db.execute('UPDATE attempts SET state=?,metric=?,failure=? WHERE id=?',(state,metric,failure,identity))

    def rows(self):
        return self.db.execute('SELECT id,configuration,state,metric,failure FROM attempts ORDER BY id').fetchall()

    def bounded_retry(self,original_identity,maximum_attempts=2):
        """A new attempt preserves the failed parent and its compute charge."""
        original=self.get(original_identity)
        if original is None:raise ValueError('Original attempt absent')
        configuration=json.loads(original[1])
        if 'retry_parent' in configuration:raise ValueError('Retry must identify original recipe')
        related=[r for r in self.rows() if json.loads(r[1]).get('retry_parent')==original_identity]
        related.sort(key=lambda row:json.loads(row[1])['attempt_index'])
        latest=related[-1] if related else original
        if latest[2] in {'PLANNED','RUNNING','SUCCEEDED'}:return latest[0]
        if latest[2] not in {'FAILED_RETRYABLE','INTERRUPTED','PAUSED_BUDGET'}:
            raise RuntimeError('Permanent/pruned attempt cannot automatically retry')
        attempt_index=len(related)+1
        if attempt_index>=maximum_attempts:raise RuntimeError('Bounded retry ceiling reached; diagnose before further attempts')
        return self.register({**configuration,'retry_parent':original_identity,'attempt_index':attempt_index})

    def get(self,identity):
        return self.db.execute('SELECT id,configuration,state,metric,failure FROM attempts WHERE id=?',(identity,)).fetchone()

    def close(self):
        self.db.close()


class ComputeBudget:
    """Persistent conservative GPU wall-time charges, including failures/crashes.

    A reservation is charged before work. Normal completion replaces it with
    measured elapsed time; a killed process keeps its full reservation. Resume
    never resets consumption, and reservation identity cannot be reused.
    """
    def __init__(self, path, ceiling):
        self.ledger = TrialLedger(path)
        self.db = self.ledger.db
        self.ceiling = float(ceiling)
        if not self.ceiling > 0:
            raise ValueError('Positive fixed ceiling required')
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS budget_settings (id INTEGER PRIMARY KEY, ceiling REAL NOT NULL)')
            self.db.execute('INSERT OR IGNORE INTO budget_settings VALUES (1,?)', (self.ceiling,))
            stored = self.db.execute('SELECT ceiling FROM budget_settings WHERE id=1').fetchone()[0]
            if stored != self.ceiling:
                raise PermissionError('Automatic budget change denied')
            self.db.execute('CREATE TABLE IF NOT EXISTS compute_charges (id TEXT PRIMARY KEY, seconds REAL NOT NULL, state TEXT NOT NULL)')

    @property
    def remaining(self):
        consumed = self.db.execute('SELECT COALESCE(SUM(seconds),0) FROM compute_charges').fetchone()[0]
        return max(0., self.ceiling-consumed)

    @contextmanager
    def charge(self, identity, reservation):
        if not reservation > 0:
            raise ValueError('Positive reservation required')
        # IMMEDIATE makes check+reservation indivisible across clients.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT id FROM compute_charges WHERE id=?', (identity,)).fetchone():
                raise ValueError('Compute attempt already charged; new retry identity required')
            if reservation > self.remaining:
                raise RuntimeError('PAUSED_BUDGET: persistent allowance insufficient')
            self.db.execute('INSERT INTO compute_charges VALUES (?,?,?)', (identity, reservation, 'RESERVED'))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        start = time.monotonic()
        state = 'FAILED'
        try:
            yield
            state = 'SUCCEEDED'
        finally:
            elapsed = time.monotonic()-start
            with self.db:
                self.db.execute('UPDATE compute_charges SET seconds=?,state=? WHERE id=?', (elapsed, state, identity))

    def close(self):
        self.ledger.close()


def measured_bill(pilot_seconds,run_count,remaining_gpu_seconds,peak_vram,free_vram):
    if not pilot_seconds or pilot_seconds<=0 or run_count<1:
        raise ValueError('Measured real workload pilot and positive run count required')
    expected=pilot_seconds*run_count
    return {'expected_gpu_seconds':expected,'run_count':run_count,
            'state':'READY' if expected<=remaining_gpu_seconds and peak_vram<=free_vram-1024**3 else 'PAUSED_BUDGET'}


def validate_architecture_contrast(left,right):
    for key in ['track','price_mode','pit_tier','grid_hash','normalizer_id','information_hash','context','horizon']:
        if key not in left or left[key]!=right.get(key):
            raise ValueError('Architecture contrast mismatch: '+key)
    return True


def eligible_candidate(experiment,partition):
    if experiment in {'E04','E07','E08','E10'} and partition=='reserved':
        raise PermissionError('Development-only experiment cannot access final')
    return experiment!='E04'
