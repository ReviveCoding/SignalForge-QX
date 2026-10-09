"""Final expanded CPU-only scope; retains initial85-pass receipt."""
from .source import *
import pytest,xml.etree.ElementTree as ET
if __name__=='__main__':
 paths=['model_risk_v38/test_model_risk_v38.py','model_risk_v38/test_supplement.py','diagnostics_v3/test_model_risk.py']
 code=pytest.main(paths+['-q','--import-mode=importlib','-p','no:cacheprovider',f'--junitxml={OUT}/tests_final_junit.xml'])
 suites=ET.parse(OUT/'tests_final_junit.xml').getroot();counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']};perfile={}
 for suite in suites:
  for case in suite:
   if case.tag=='testcase':name=case.attrib.get('classname','');perfile[name]=perfile.get(name,0)+1
 writej('tests_final.json',{'exit_code':int(code),'counts':counts,'per_file_test_cases':perfile,'test_paths':paths,'test_code_hashes':{p:sha(R/p) for p in paths},'prior_scope':'24 pure saved-diagnostics unit cases, no original ledger test calls','synthetic':'TEST_ONLY; no whole-repository success claim','initial_receipt_retained':'tests.json (85pass)'})
 progress('FINAL_TESTS_PASS' if not code else 'FINAL_TESTS_FAILED',{'counts':counts,'per_file':perfile});raise SystemExit(code)
