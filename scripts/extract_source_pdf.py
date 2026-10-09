"""Isolated source-PDF extraction and visible provenance, no research-env changes."""
import json,subprocess,sys
from signalforge.runtime import paths,commit_bundle
from signalforge.sources import Acquisition
repo,runtime=paths();url=sys.argv[1];stem=sys.argv[2]
if not stem.replace('_','').isalnum():raise ValueError('Safe output stem required')
client=Acquisition(runtime,2*1024**2);raw=client.cached(url,'pdf') or client.fetch(url,'pdf')
output=repo/'reports/source_qa'/(stem+'.png');output.parent.mkdir(exist_ok=True)
code='''import fitz,sys,json
from pypdf import PdfReader
pdf=fitz.open(sys.argv[1]);pdf[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(sys.argv[2])
page=PdfReader(sys.argv[1]).pages[0]
print(json.dumps(dict(text=page.extract_text(),layout=page.extract_text(extraction_mode='layout'))))
'''
result=subprocess.run([str(runtime/'tools/pdf_qa_env/bin/python'),'-c',code,raw['path'],str(output)],capture_output=True,text=True,check=True,timeout=30)
extraction=json.loads(result.stdout)
commit_bundle(runtime/'artifacts/source_pdf_extractions'/raw['sha256'],{'extraction.json':extraction},
              {'source_sha256':raw['sha256'],'tools':'isolated PyMuPDF1.25.5+pypdf5.4.0'})
print(extraction['layout'][:4000])
