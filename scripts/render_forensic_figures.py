"""Standalone figures of all immutable development forecasts, no fit selection."""
import json,io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from signalforge.runtime import paths,validate_bundle,commit_bundle,digest,file_hash,atomic_bytes,atomic_json,now
repo,runtime=paths();receipt=json.loads((repo/'reports/track_model_forensics.json').read_text());folder=runtime/receipt['relative_bundle'];validate_bundle(folder)
table=pd.read_parquet(folder/'family_information.parquet');rows=pd.read_parquet(folder/'rows.parquet');outputs={}
fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
for axis,track in zip(axes,['Main-A','Nested-B']):
    values=table[table.track.eq(track)].pivot(index='family',columns='information',values='normalized_pinball').sort_index()
    image=axis.imshow(values.to_numpy(),aspect='auto',norm=LogNorm(vmin=.15,vmax=max(table.normalized_pinball)),cmap='viridis')
    axis.set_xticks(range(len(values.columns)),values.columns);axis.set_yticks(range(len(values)),values.index);axis.set_title(track)
    for (i,j),v in np.ndenumerate(values.to_numpy()):axis.text(j,i,f'{v:.3g}',ha='center',va='center',fontsize=8,color='white' if v<5 else 'black')
fig.colorbar(image,ax=axes,label='Normalized pinball, logarithmic colors; lower is better')
fig.suptitle('Immutable P0 development: assets then seeds averaged within date, then dates\nRaw unadjusted price ratios include split jumps; no final/P1 qualification')
buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=150);outputs['forensic_information_pinball.png']=buffer.getvalue();plt.close(fig)
fig,axis=plt.subplots(figsize=(9,6),layout='constrained')
image=axis.hexbin(np.log10(1+rows.transformed_abs_max),np.log10(1+rows.forecast_abs_max),gridsize=60,mincnt=1,bins='log')
fig.colorbar(image,label='Forecast-row count per bin, logarithmic colors')
axis.set(xlabel='log10(1 + maximum absolute transformed context)',ylabel='log10(1 + maximum absolute quantile forecast)',title=f'All {len(rows):,} immutable forecast rows retained\nPost-hoc association, not an intervention or causal proof')
buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=150);outputs['forensic_extrapolation.png']=buffer.getvalue();plt.close(fig)
identity=digest({'source':file_hash(repo/'reports/track_model_forensics.json'),'figures':{k:digest(v.hex()) for k,v in outputs.items()}});target=runtime/'artifacts/forensic_figures'/identity
manifest={'state':'RENDERED_ALL_ROW_DEVELOPMENT_FORENSIC_FIGURES','created_at':now(),'source_receipt_sha256':file_hash(repo/'reports/track_model_forensics.json'),'forecast_rows':len(rows),'relative_bundle':str(target.relative_to(runtime)),'reserved_access':False,'qualified_for_final':False,'figures':{k:__import__('hashlib').sha256(v).hexdigest() for k,v in outputs.items()}}
commit_bundle(target,{**outputs,'manifest.json':manifest},{'reserved_access':False})
for name,data in outputs.items():atomic_bytes(repo/'reports/figures/forensic_snapshots'/identity/name,data)
manifest['repo_snapshot_directory']='reports/figures/forensic_snapshots/'+identity;atomic_json(repo/'reports/forensic_figures.json',manifest);print(json.dumps(manifest,indent=2))
