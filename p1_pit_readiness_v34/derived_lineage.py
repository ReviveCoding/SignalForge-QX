"""Verify INDPRO derived YoY lineage against actual raw FRED parent bytes."""
import json,re
from pathlib import Path
import pandas as pd
from signalforge.runtime import digest,file_hash,atomic_json
R=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX-v33-dev');O=Path('/mnt/c/Users/USERNAME/Downloads/SignalForge-QX');T=Path('/home/USERNAME/.local/share/signalforge-qx-v3/da768f7446b1');D=R/'reports/p1_pit_v34'
def main():
    card=next(s for s in json.loads((O/'reports/strict_pit_final_gate_audit.json').read_text())['sources'] if s['source']=='macro');assert file_hash(T/card['relative_path'])==card['sha256'];events=json.loads((T/card['relative_path']).read_text());derived=[r for r in events if r['entity']=='INDPRO'];parents=set();checks=[]
    for row in derived:
        deps=sorted(row['dependency_raw_hashes']);period=str(pd.Timestamp(row['reference_time']).tz_convert('UTC').tz_localize(None).to_period('M'));expected=digest({'transform':'same_vintage_yoy_percent_v1','series':'INDPRO','reference_month':period,'vintage':row['fred_vintage_date'],'raw_hashes':deps});assert expected==row['raw_hash'];parents.update(deps);checks.append({'lineage_hash':expected,'parents':deps,'reference_month':period,'vintage':row['fred_vintage_date'],'unit':row['unit']});assert row['unit']=='percent_yoy'
    index={p.stem:p for p in (T/'raw').rglob('*') if p.is_file() and p.stem in parents};verified={}
    for sha in sorted(parents):
        assert sha in index,'Missing explicit INDPRO raw ancestor';assert file_hash(index[sha])==sha;verified[sha]=str(index[sha])
    atomic_json(D/'INDPRO_derived_lineage_validation.json',{'passed':True,'derived_versions':len(derived),'unique_lineage_ids':len({r['lineage_hash'] for r in checks}),'raw_parent_hashes_verified':len(verified),'raw_parent_files':verified,'checks':checks,'same_vintage_ratio_semantics_preserved':True,'first_publication_qualified':False,'reserved_access':False});print(json.dumps({'INDPRO_derived_versions':len(derived),'unique_lineage_ids':len({r['lineage_hash'] for r in checks}),'raw_parents_verified':len(verified)}),flush=True)
if __name__=='__main__':main()