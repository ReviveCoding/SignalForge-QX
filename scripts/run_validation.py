"""Run actual tests and persist execution evidence and acceptance mappings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re
import xml.etree.ElementTree as ET
import argparse
from signalforge.runtime import atomic_json,paths,now,file_hash,code_hash

repo,runtime=paths()
parser=argparse.ArgumentParser();parser.add_argument('--cpu-only',action='store_true');parser.add_argument('--real-source',action='store_true');options=parser.parse_args()
folder=runtime/'artifacts/validation'/now().replace(':','').replace('+','_');folder.mkdir(parents=True,exist_ok=True)
results=[]
source_hash=code_hash(repo)
qualification_path=repo/'reports/track_input_qualification_v311.json'
qualification_hash=file_hash(qualification_path) if qualification_path.exists() else None
suites=[('handoff',['-m','unittest','discover','-s','tests/handoff','-v']),
                   ('unit',['-m','pytest','tests/unit','-q','--junitxml='+str(folder/'unit.xml')]),
                   ('gpu',['-m','pytest','tests/gpu','-q','--junitxml='+str(folder/'gpu.xml')])]
if options.real_source:suites.append(('real_source',['-m','pytest','tests/real','-q','--junitxml='+str(folder/'real_source.xml')]))
for suite,args in suites:
    if options.cpu_only and suite=='gpu':continue
    t=time.monotonic();log=folder/(suite+'.log')
    with log.open('w') as f:
        r=subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,timeout=300)
    if suite=='handoff':
        match=re.search(r'Ran (\d+) tests? in',log.read_text())
        counts={'tests':int(match[1]) if match else 0,'failures':0 if r.returncode==0 else None,'errors':0 if r.returncode==0 else None}
        execution=[];test_files={}
    else:
        root=ET.parse(folder/(suite+'.xml')).getroot()
        counts={key:sum(int(s.attrib.get(key,0)) for s in root.iter('testsuite')) for key in ['tests','failures','errors','skipped']}
        execution=[];test_files={}
        for test in root.iter('testcase'):
            source=test.attrib['classname'].replace('.','/')+'.py'
            function=test.attrib['name'].split('[')[0]
            execution.append({'identity':source+'::'+function,'parameter_case':test.attrib['name'],
                              'state':'FAILED' if test.find('failure') is not None or test.find('error') is not None else 'SKIPPED' if test.find('skipped') is not None else 'PASSED'})
            if (repo/source).is_file():test_files[source]=file_hash(repo/source)
    results.append({'suite':suite,'exit_code':r.returncode,'counts':counts,'seconds':time.monotonic()-t,
                    'log':str(log),'log_sha256':file_hash(log),'evidence_kind':'real_cached_source_and_artifact_smoke' if suite=='real_source' else 'synthetic_fixture',
                    'executed_cases':execution,'test_file_hashes':test_files,'source_tree_hash':source_hash})
    print(json.dumps({k:v for k,v in results[-1].items() if k not in {'executed_cases','test_file_hashes'}}),flush=True)
atomic_json(repo/'reports'/('cpu_test_execution.json' if options.cpu_only else 'test_execution.json'),
            {'created_at':now(),'results':results,'full_acceptance_complete':False,
             'qualification_sha256':qualification_hash,
             'inputs_unchanged':code_hash(repo)==source_hash and qualification_hash==(file_hash(qualification_path) if qualification_path.exists() else None),
             'suite_scope':'CPU only; prior GPU evidence is separate' if options.cpu_only else 'full serial handoff/unit/CUDA'})
raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)
