"""Dependency-free historical G17 HTML recovery; preserves initial parser incidents."""
from html.parser import HTMLParser
from data_recovery_v36.acquire import R,O,load,append_progress
from data_recovery_v36.runner import write
from qualification_v35.core import sha
from pathlib import Path
class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.depth=0;self.table=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':
            self.depth+=1
            if self.depth==1:self.table=[]
        if self.depth==1 and tag=='tr':self.row=[]
        if self.depth==1 and tag in ['td','th']:self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if self.depth==1 and tag in ['td','th'] and self.cell is not None:
            if self.row is not None:self.row.append(' '.join(' '.join(self.cell).split()))
            self.cell=None
        if self.depth==1 and tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
        if tag=='table':
            if self.depth==1:self.tables.append(self.table)
            self.depth-=1

def recover():
    records=load(O/'attempted_public_sources.json')['records'];out=[]
    for r in records:
        if r['family']!='FED_G17' or not r.get('payload_path'):continue
        b=Path(r['payload_path']).read_bytes();assert sha(b)==r['sha256'];p=Tables();p.feed(b.decode());assert p.depth==0 and len(p.tables)>=2
        # Tables remain verbatim cells: no inferred reference date / base or vintage relabeling.
        x={'url':r['url'],'payload_sha256':r['sha256'],'parser':'stdlib.HTMLParser','tables':p.tables,'table_count':len(p.tables),'table_rows':sum(len(t) for t in p.tables),'first_public_qualified':False,'canonical_join':'BLOCKED_NO_SCOPED_EXPORT','numeric_base_not_relabelled':True};out.append(x)
    write('G17_table_recovery_v2.json',out);append_progress('G17_PARSER_RECOVERY_PASS',documents=len(out),table_rows=sum(x['table_rows'] for x in out),original_incidents_preserved=True);return out
if __name__=='__main__':recover()
