import json
from pathlib import Path
from signalforge.runtime import paths
repo,runtime=paths()
for p in (runtime/'raw').glob('*/*.receipt.json'):
    d=json.loads(p.read_text())
    if d.get('metadata',{}).get('series_id'):
        b=json.loads(Path(d['path']).read_text())
        obs=b.get('observations',[]); cells=sum(max(0,len(r)-1) for r in obs)
        print(d['metadata'].get('series_id'),d['metadata'].get('page'),b.get('offset'),b.get('limit'),b.get('count'),len(obs),cells,b.get('output_type'),b.get('realtime_start'),b.get('realtime_end'))
