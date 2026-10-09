"""Execute available registered CPU Main/Nested branches; retain source blockers."""
import argparse,json
from signalforge.runtime import paths,atomic_json
from signalforge.track_inputs import qualify_track_inputs
from signalforge.track_engine import run_panel_development,CPU_FAMILIES

p=argparse.ArgumentParser()
p.add_argument('--track',choices=['Main-A','Nested-B','both'],default='both')
p.add_argument('--input-version',choices=['auto','base','v31','v311'],default='auto')
a=p.parse_args()
repo,runtime=paths();results=[]
tracks=['Main-A','Nested-B'] if a.track=='both' else [a.track]
if a.input_version=='auto':
    version='v311' if all((repo/'.local'/('inputs_'+track+'_v311.json')).exists() for track in tracks) else 'base'
else:version=a.input_version

for track in tracks:
    try:
        panels,contexts,qualification=qualify_track_inputs(repo,runtime,track,version=version)
        qualification['input_version']=version
        qualification['manifest_path']=str(repo/'.local'/('inputs_'+track+('' if version=='base' else '_'+version)+'.json'))
        if panels is not None:
            study=json.loads((repo/'configs/study.json').read_text())
            families=['historical','ewma','ridge','linear_quantile','mixed_frequency_shrinkage']
            if not set(families)<=CPU_FAMILIES:
                raise PermissionError('CPU track runner cannot launch CUDA families')
            qualification['cpu_development']=run_panel_development(
                repo,runtime,panels,contexts,study,track,families,
                budget=None,evidence_kind='public_data_reconstructed')
    except (ValueError,RuntimeError,PermissionError,FileNotFoundError,KeyError,TypeError) as error:
        qualification={'track':track,'input_version':version,
            'state':'PAUSED_BUDGET' if str(error).startswith('PAUSED_BUDGET') else 'BLOCKED_INPUT_QUALIFICATION',
            'reason':str(error),'qualified_for_final':False,'reserved_access':False}
    from signalforge.track_planning import track_cuda_pilot_boundary
    qualification['cuda_development']=track_cuda_pilot_boundary(repo)
    results.append(qualification)

result={'state':'PARTIAL_CPU_DEVELOPMENT_CUDA_BLOCKED' if any('cpu_development' in r for r in results) else 'BLOCKED_DATA',
    'input_version':version,'tracks':results,'reserved_access':False,'economic_qualified':False}
if a.track=='both':
    atomic_json(repo/'reports/track_input_qualification.json',result)
    if version!='base':
        atomic_json(repo/'reports'/('track_input_qualification_'+version+'.json'),result)
else:
    safe=a.track.replace('-','_')
    atomic_json(repo/'reports'/('track_input_qualification_'+version+'_'+safe+'.json'),result)
print(json.dumps(result,indent=2))
