"""Read only raw bytes explicitly referenced by permitted development events."""
import json,re
from pathlib import Path
import pandas as pd
from signalforge.runtime import file_hash,atomic_json
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');T=Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1');O=R/'reports/p1_pit_v34'
def main():
    events=pd.read_csv(O/'per_version_clock_audit.csv');refs=set(events.payload_sha256.dropna());index={}
    for p in (T/'raw').rglob('*'):
        if p.is_file() and re.fullmatch('[0-9a-f]{64}',p.stem):index.setdefault(p.stem,[]).append(p)
    rows=[]
    for sha in sorted(refs):
        paths=index.get(sha,[]);checked=[]
        for p in paths:
            actual=file_hash(p);assert actual==sha,'Referenced immutable source raw corruption';checked.append(str(p))
        rows.append({'raw_sha256':sha,'referenced_event_rows':int(events.payload_sha256.eq(sha).sum()),'source_types':','.join(sorted(set(events[events.payload_sha256.eq(sha)].source))),'actual_raw_byte_files_verified':len(checked),'paths':checked,'state':'RAW_BYTES_HASH_VERIFIED_NO_FIRST_CLOCK_PROOF' if checked else 'NO_DIRECT_RAW_FILE_OR_DERIVED_REFERENCE_REQUIRE_DEPENDENCY_RECEIPT'})
    atomic_json(O/'raw_dependency_validation.json',{'references':len(rows),'verified_raw_hashes':sum(r['actual_raw_byte_files_verified']>0 for r in rows),'without_direct_raw_file':sum(r['actual_raw_byte_files_verified']==0 for r in rows),'rows':rows,'reserved_files_opened':False,'first_publication_qualified':False});print(json.dumps({'raw_references':len(rows),'verified':sum(r['actual_raw_byte_files_verified']>0 for r in rows),'not_directly_resolved':sum(r['actual_raw_byte_files_verified']==0 for r in rows)}),flush=True)
if __name__=='__main__':main()