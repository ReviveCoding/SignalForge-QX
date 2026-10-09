"""Standalone scientific figures from captured development diagnostics only."""
import json,io,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from signalforge.runtime import paths,atomic_json,atomic_bytes,commit_bundle,digest,now
repo,runtime=paths();source=repo/'reports/auxiliary_development_analysis.json';content=source.read_bytes();analysis=json.loads(content)
if analysis.get('state')!='SUCCEEDED_DIAGNOSTIC' or analysis.get('final_access') is not False:raise RuntimeError('Completed development diagnostic required')
table=analysis['comparison_table'];families=[r['family'] for r in table if r['state']=='SUCCEEDED_DIAGNOSTIC']
if not families:raise RuntimeError('No development scores to plot')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
plt.rcParams['svg.hashsalt']='SignalForge-QX-development-v3'
outputs={}
def save(name,figure):
    for suffix in ['png','svg']:
        stream=io.BytesIO();figure.savefig(stream,format=suffix,bbox_inches='tight',**({'metadata':{'Date':None}} if suffix=='svg' else {}));outputs[name+'.'+suffix]=stream.getvalue()
    plt.close(figure)
fig,axes=plt.subplots(1,2,figsize=(11,max(5,.38*len(families))),gridspec_kw={'width_ratios':[1.5,1]})
axes[0].barh(families,[r['pinball'] for r in table],color='#346b9e');axes[0].invert_yaxis()
axes[0].set_xlabel('Train-scale normalized multi-quantile pinball (lower is better)');axes[0].set_xlim(left=0)
axes[1].scatter([r['interval_90_coverage'] for r in table],families,color='#346b9e');axes[1].axvline(.9,color='#b95132',linestyle='--',label='nominal 90%')
axes[1].set_xlim(0,1);axes[1].invert_yaxis();axes[1].set_xlabel('Observed 90% interval coverage');axes[1].legend(loc='lower left')
axes[1].tick_params(axis='y',labelleft=False)
fig.suptitle(f"Auxiliary development: {len(families)} completed families, 260 market dates\nReconstructed Tier B; no final or economic qualification")
save('development_scores_and_coverage',fig)
contrasts=[c for c in analysis['paired_contrasts'] if 'mean_improvement' in c['blocks'].get('8',{})]
if contrasts:
    fig,axis=plt.subplots(figsize=(9,max(3,len(contrasts)*.3)))
    for i,c in enumerate(contrasts):
        r=c['blocks']['8'];axis.plot([r['lower_95'],r['upper_95']],[i,i],color='#346b9e');axis.scatter([r['mean_improvement']],[i],color='#346b9e')
    axis.set_yticks(range(len(contrasts)),[c['contrast'] for c in contrasts]);axis.invert_yaxis();axis.axvline(0,color='#555',linewidth=1)
    axis.set_xlabel('Reference minus candidate normalized loss; positive favors candidate')
    axis.set_title('Exploratory paired 8-week block intervals\nDates are resampling units; seeds are not market samples')
    save('development_paired_intervals',fig)
family='rgmf_gru' if 'rgmf_gru' in families else 'ridge'
rows=analysis['forecast_rows'][family]
if any(pd.Timestamp(r['decision_time'])>=pd.Timestamp('2024-01-01T00:00Z') for r in rows):raise PermissionError('Reserved forecast plot forbidden')
data=pd.DataFrame(rows).sort_values('decision_time');data.index=pd.to_datetime(data.decision_time,utc=True)
# Gaps are drawn as gaps rather than suggesting forecasts across missing dates.
calendar=pd.date_range(data.index.min().normalize(),data.index.max().normalize(),freq='W-FRI',tz='UTC')
data.index=data.index.normalize();data=data.reindex(calendar)
q=np.array([v if isinstance(v,list) else [np.nan]*5 for v in data.quantiles])
fig,axis=plt.subplots(figsize=(11,3.5));axis.fill_between(calendar,q[:,0],q[:,-1],color='#9dbad1',alpha=.45,label='raw ensemble 90% interval')
axis.plot(calendar,data.y,color='#444',linewidth=.8,label='next-release observed inventory change')
axis.plot(calendar,data['mean'],color='#346b9e',linewidth=.8,label='separate mean head')
axis.set_ylabel('Million barrels');axis.set_title(f'{family}: next-release inventory forecasts, 2018–2022\nHistorical raw forecasts; no causal/return or prospective claim');axis.legend(ncol=3,fontsize=8)
save('inventory_forecasts',fig)
manifest={'created_at':now(),'state':'RENDERED_DEVELOPMENT_FIGURES','source_sha256':hashlib.sha256(content).hexdigest(),'families':families,
          'source_path':str(source),'qualified_for_final':False,'reserved_access':False,'figures':{name:hashlib.sha256(value).hexdigest() for name,value in outputs.items()}}
identity=digest({'source':manifest['source_sha256'],'figures':manifest['figures']})
manifest['repo_snapshot_directory']='reports/figures/snapshots/'+identity
commit_bundle(runtime/'artifacts/research_figures'/identity,{**outputs,'analysis.json':content,'manifest.json':manifest},{'source_sha256':manifest['source_sha256'],'reserved_access':False})
for name,value in outputs.items():
    atomic_bytes(repo/'reports/figures'/name,value)
    atomic_bytes(repo/manifest['repo_snapshot_directory']/name,value)
atomic_json(repo/'reports/research_figures.json',{**manifest,'immutable_artifact_id':identity});print(identity,len(outputs))
