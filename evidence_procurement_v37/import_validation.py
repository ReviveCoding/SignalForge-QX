"""User-invoked private inbox validation; never downloads or reads a rejected dataset."""
import argparse,json,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
from evidence_procurement_v37.core import accept,register_import

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',required=True);p.add_argument('--manifest',required=True);p.add_argument('--registry',default='evidence_procurement_v37/trust_registry.json');p.add_argument('--inbox',required=True);p.add_argument('--receipt-dir',required=True);p.add_argument('--test-only',action='store_true');p.add_argument('--record-db');a=p.parse_args()
    destination=Path(a.receipt_dir).resolve();root=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev/reports/evidence_procurement_v37').resolve();runtime=Path('/home/USERNAME/.local/share/signalforge-qx-v33-dev/isolated-engineering/evidence_procurement_v37').resolve()
    if not destination.is_relative_to(root) and not destination.is_relative_to(runtime):raise PermissionError('New v37 receipts only')
    # Control envelopes/independently enrolled trust metadata only; payload not opened here.
    try:
        m=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'));reg=json.loads(Path(a.registry).read_text(encoding='utf-8-sig'));value=accept(a.payload,m,reg,a.inbox,test_only=a.test_only);out={k:v for k,v in value.items() if k!='records'}
        if a.record_db:
            db=Path(a.record_db).resolve()
            if not db.is_relative_to(runtime):raise PermissionError('New v37 ext4 import ledger only')
            out['import_receipt']=register_import(db,value)
        exit_code=0
    except (PermissionError,ValueError,KeyError,TypeError,OSError) as error:
        out={'state':'REJECTED_FAIL_CLOSED','error_type':type(error).__name__,'reason':str(error) if isinstance(error,(ValueError,PermissionError)) and not isinstance(error,json.JSONDecodeError) else 'Invalid/unavailable control or supplied input','qualified_Tier_A':0,'P1_qualified_sessions':0,'real_forward_allowed':False};exit_code=2
    out['observed_utc']=datetime.now(timezone.utc).isoformat();destination.mkdir(parents=True,exist_ok=True);name=destination/('import_'+uuid.uuid4().hex+'.json')
    with name.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({**out,'receipt':str(name)}));return exit_code
if __name__=='__main__':sys.exit(main())
