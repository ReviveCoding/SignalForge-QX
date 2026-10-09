"""Local append-only TEST_ONLY shadow-readiness recorder. Real issuance always forbidden."""
import json,sqlite3,sys,math,argparse
from pathlib import Path
from datetime import datetime,timezone
from qualification_v35.core import canonical,sha,clock,valid_sha
HASH_KEYS=['model_hash','source_hash','normalizer_hash','benchmark_hash','target_hash']
FIELDS=set(['kind','retry_key','track','asset','seed','decision_at','source_valid_at','source_known_at','label_end','label_available_at','expiry_at','quantiles','block',*HASH_KEYS])
def actual_utc():return datetime.now(timezone.utc).isoformat()
def validate_message(m,plan):
    if m.get('kind')!='TEST_ONLY':raise PermissionError('V36 blocks ALL real market/forward issuance before any DB write')
    if set(m)!=FIELDS:raise ValueError('Strict shadow schema')
    if not m['retry_key'] or m['track'] not in ['Main-A','Nested-B'] or m['asset'] not in plan['assets'] or m['seed'] not in plan['seeds']:raise ValueError('Registered cohort identity')
    if isinstance(m['block'],bool) or m['block'] not in [0,1]:raise ValueError('Two registered cohorts')
    for k in HASH_KEYS:
        if not valid_sha(m[k]) or m[k] not in plan[k+'s']:raise ValueError('Exact frozen '+k)
    if plan.get('track_bindings'):
        bound=plan['track_bindings'][m['track']]
        for key in ['model_hash','source_hash','normalizer_hash','target_hash','benchmark_hash']:
            if m[key] not in bound[key+'s']:raise ValueError('Track-specific frozen binding')
    d,v,k,end,available,expiry=map(clock,[m['decision_at'],m['source_valid_at'],m['source_known_at'],m['label_end'],m['label_available_at'],m['expiry_at']])
    if not v<=k<=d<expiry<=end<=available:raise ValueError('Source/expiry/maturity ordering')
    q=m['quantiles']
    if len(q)!=5 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in q) or q!=sorted(q):raise ValueError('Finite ordered five quantiles')
    return {'fixture_only':True,'semantic_identity':sha(canonical([m['track'],d.isoformat(),m['asset'],m['seed']])),'message_sha256':sha(canonical(m))}

def replay(path):
    with sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT id,retry_key,semantic_identity,message_sha256,observed_utc,body,previous_sha256,receipt_sha256 FROM records ORDER BY id').fetchall()
    previous='0'*64;last=None
    for expected,row in enumerate(rows,1):
        i,key,identity,msg,at,body,prev,receipt=row
        if i!=expected or prev!=previous or sha(body.encode())!=msg:raise ValueError('Append/hash chain integrity')
        if json.loads(body).get('kind')!='TEST_ONLY':raise ValueError('Real payload cannot be v36 readiness')
        t=clock(at)
        if last and t<last:raise ValueError('Wall-clock regression')
        content={'id':i,'retry_key':key,'semantic_identity':identity,'message_sha256':msg,'observed_utc':at,'previous_sha256':prev,'state':'TEST_ONLY_READINESS_NOT_FORECAST'}
        if sha(canonical(content))!=receipt:raise ValueError('Receipt identity mismatch')
        previous=receipt;last=t
    return {'records':len(rows),'head_sha256':previous,'real_forecasts':0,'state':'TEST_ONLY_REPLAY_PASS'}

class ShadowRecorder:
    def __init__(self,path,plan,clock_fn=actual_utc):self.path=Path(path);self.plan=plan;self.clock_fn=clock_fn
    def append(self,message):
        check=validate_message(message,self.plan) # no output on gate failure
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(self.path,timeout=10)
        try:
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY,retry_key TEXT UNIQUE NOT NULL,semantic_identity TEXT UNIQUE NOT NULL,message_sha256 TEXT NOT NULL,observed_utc TEXT NOT NULL,body TEXT NOT NULL,previous_sha256 TEXT NOT NULL,receipt_sha256 TEXT UNIQUE NOT NULL)')
            db.execute("CREATE TRIGGER IF NOT EXISTS deny_update BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT,'Append-only'); END")
            db.execute("CREATE TRIGGER IF NOT EXISTS deny_delete BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT,'Append-only'); END")
            db.execute('BEGIN IMMEDIATE');existing=db.execute('SELECT id,message_sha256,receipt_sha256 FROM records WHERE retry_key=?',(message['retry_key'],)).fetchone()
            if existing:
                if existing[1]!=check['message_sha256']:raise ValueError('Retry identity conflict')
                db.rollback();return {'id':existing[0],'receipt_sha256':existing[2],'reused':True,'real_forecast':False}
            head=db.execute('SELECT id,receipt_sha256,observed_utc FROM records ORDER BY id DESC LIMIT 1').fetchone();observed=self.clock_fn();t=clock(observed)
            if head and t<clock(head[2]):raise ValueError('Actual wall-clock regressed; do not backdate')
            ident=(head[0]+1) if head else 1;previous=head[1] if head else '0'*64;content={'id':ident,'retry_key':message['retry_key'],'semantic_identity':check['semantic_identity'],'message_sha256':check['message_sha256'],'observed_utc':observed,'previous_sha256':previous,'state':'TEST_ONLY_READINESS_NOT_FORECAST'};receipt=sha(canonical(content));db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',(ident,message['retry_key'],check['semantic_identity'],check['message_sha256'],observed,canonical(message).decode(),previous,receipt));db.commit();return {**content,'receipt_sha256':receipt,'reused':False,'real_forecast':False}
        finally:db.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['replay','append-test-only'],required=True);p.add_argument('--database',required=True);p.add_argument('--plan');p.add_argument('--input');a=p.parse_args()
    if a.action=='replay':print(json.dumps(replay(a.database)));return
    if not a.plan or not a.input:raise ValueError('Explicit fixture plan and TEST_ONLY input')
    plan=json.loads(Path(a.plan).read_text());message=json.loads(Path(a.input).read_text());print(json.dumps(ShadowRecorder(a.database,plan).append(message)))
if __name__=='__main__':main()
