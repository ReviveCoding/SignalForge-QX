import json,time
from .source import R,OUT,sha,writej,progress
import pytest
if __name__=='__main__':
 paths=['model_risk_v38/test_model_risk_v38.py','diagnostics_v3/test_model_risk.py']
 args=paths+['-q','--import-mode=importlib','-p','no:cacheprovider',f'--junitxml={OUT}/tests_junit.xml']
 code=pytest.main(args)
 import xml.etree.ElementTree as ET
 suites=ET.parse(OUT/'tests_junit.xml').getroot();counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
 writej('tests.json',{'exit_code':int(code),'counts':counts,'test_paths':paths,'new_test_code_sha256':sha(R/paths[0]),'prior_tests':'scoped pure diagnostics core; not whole project or historical105pass/1fail receipt replacement','synthetic':'TEST_ONLY','scope':'CPU only, no original ledger readers'})
 progress('TESTS_PASS' if code==0 else 'TESTS_FAILED',counts)
 raise SystemExit(code)
