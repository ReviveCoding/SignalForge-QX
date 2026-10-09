from pathlib import Path
from signalforge.runtime import paths
repo,runtime=paths()
roots=['track_source_inputs_v311','macro_canonical_v311','cftc_canonical','eia_market_canonical','nport_canonical_v311']
for name in roots:
    print('===',name)
    root=runtime/'artifacts'/name
    if not root.exists():
        print('ABSENT');continue
    files=sorted((p for p in root.rglob('*') if p.is_file()),key=lambda p:p.stat().st_size,reverse=True)
    for p in files[:8]:
        print(p.stat().st_size, p.relative_to(runtime))
