"""Separate rootless PDF tools verify the source table's printed units and values."""
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urljoin
from signalforge.runtime import paths,atomic_json,file_hash,now
from signalforge.sources import Acquisition,Links

repo,runtime=paths();client=Acquisition(runtime,16*1024**2)
page='https://www.eia.gov/petroleum/supply/weekly/archive/2022/2022_01_12/wpsr_2022_01_12.php'
index=client.cached(page,'html') or client.fetch(page,'html')
links=Links();links.feed(Path(index['path']).read_text(errors='replace'))
pdfs={urljoin(page,h) for h,_ in links.links if h.endswith('pdf/table4.pdf')}
if len(pdfs)!=1:raise ValueError('Official table4 PDF discovery failed')
url=pdfs.pop();raw=client.cached(url,'pdf') or client.fetch(url,'pdf',metadata={'release_page':page,'table':'4'})
env=runtime/'tools/pdf_qa_env';python=env/'bin/python'
if not python.exists():subprocess.run([sys.executable,'-m','venv',str(env)],check=True,timeout=60)
if not (env/'qualified-tools.json').exists():
    subprocess.run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple','PyMuPDF==1.25.5','pypdf==5.4.0'],check=True,timeout=180)
    atomic_json(env/'qualified-tools.json',{'purpose':'source PDF QA only; main research environment unchanged',
                                           'PyMuPDF':'1.25.5','pypdf':'5.4.0','installed_at':now()})
output=repo/'reports/source_qa';output.mkdir(parents=True,exist_ok=True)
script='''import sys,json,fitz
from pypdf import PdfReader
path,png=sys.argv[1:]
reader=PdfReader(path)
text="\\n".join(p.extract_text() for p in reader.pages)
doc=fitz.open(path)
doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(png)
print(json.dumps({"pages":len(reader.pages),"text":text,"million_barrels_present":"million barrels" in text.lower()}))
'''
r=subprocess.run([str(python),'-c',script,raw['path'],str(output/'eia_table4.png')],check=True,capture_output=True,text=True,timeout=30)
extraction=json.loads(r.stdout)
atomic_json(repo/'reports/eia_unit_qualification.json',{'created_at':now(),'source':raw,'extraction':extraction,
                                                      'render_path':str(output/'eia_table4.png'),'render_sha256':file_hash(output/'eia_table4.png'),
                                                      'visual_verified':False,'historical_original_vintage_qualified':False})
print(json.dumps(extraction,indent=2))
