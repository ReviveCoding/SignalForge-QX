import json
from pathlib import Path
import pytest
from signalforge.sources import Acquisition


class Response:
    def __init__(self,status=200,headers=None,body=b'a,b\n1,2'):
        self.status_code=status;self.headers=headers or {};self.body=body;self.url='https://www.eia.gov/x.csv';self.closed=False
    def close(self):self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def iter_content(self,size):yield self.body


def test_forbidden_redirect_not_requested(tmp_path,monkeypatch):
    c=Acquisition(tmp_path);calls=[]
    def get(url,**kwargs):
        calls.append(url);return Response(302,{'Location':'https://evil.example/steal'})
    monkeypatch.setattr(c.session,'get',get)
    with pytest.raises(ValueError):c.get('https://www.eia.gov/x')
    assert len(calls)==1


def test_permission_denial_not_retried(tmp_path,monkeypatch):
    c=Acquisition(tmp_path);calls=[]
    def get(url,**kwargs):calls.append(url);return Response(403)
    monkeypatch.setattr(c.session,'get',get)
    with pytest.raises(PermissionError):c.get('https://www.eia.gov/x')
    assert len(calls)==1


def test_partial_response_no_raw_receipt(tmp_path,monkeypatch):
    c=Acquisition(tmp_path)
    monkeypatch.setattr(c,'get',lambda *args,**kwargs:Response(headers={'Content-Length':'999'}))
    with pytest.raises(ValueError):c.fetch('https://www.eia.gov/x.csv','csv')
    assert not list((tmp_path/'raw').glob('*/*.receipt.json'))
    assert not list((tmp_path/'raw').glob('*/*.partial'))


def test_changed_raw_bytes_immutable(tmp_path,monkeypatch):
    c=Acquisition(tmp_path);bodies=[b'a,b\n1,2',b'a,b\n3,4']
    monkeypatch.setattr(c,'get',lambda *args,**kwargs:Response(body=bodies.pop(0)))
    one=c.fetch('https://www.eia.gov/x.csv','csv');two=c.fetch('https://www.eia.gov/x.csv','csv')
    assert one['sha256']!=two['sha256']
    assert Path(one['path']).read_bytes()==b'a,b\n1,2'
    with pytest.raises(ValueError):c.cached('https://www.eia.gov/x.csv','csv')


def test_download_byte_budget_counts_received_data(tmp_path,monkeypatch):
    c=Acquisition(tmp_path,max_bytes=10)
    monkeypatch.setattr(c,'get',lambda *args,**kwargs:Response(body=b'a,b\n1,2'))
    c.fetch('https://www.eia.gov/x.csv','csv')
    assert c.remaining==3
    with pytest.raises(ValueError):c.fetch('https://www.eia.gov/x.csv','csv')

def test_fred_transient_network_retry_is_bounded_and_secret_safe(tmp_path,monkeypatch):
    import requests
    import signalforge.sources as sources
    c=Acquisition(tmp_path);calls=[]
    monkeypatch.setattr(sources.time,'sleep',lambda *_:None)
    def fail(url,**kwargs):
        calls.append(url)
        raise requests.ConnectionError('secret-bearing-url-must-not-propagate')
    monkeypatch.setattr(c.session,'get',fail)
    with pytest.raises(RuntimeError,match=r'Network failure \(ConnectionError\)') as error:
        c.get('https://api.stlouisfed.org/fred/series/observations',{'series_id':'UNRATE','api_key':'a'*32})
    assert len(calls)==4
    assert 'secret-bearing' not in str(error.value)
    assert 'a'*32 not in str(error.value)

def test_non_fred_network_retry_remains_two_attempts(tmp_path,monkeypatch):
    import requests
    import signalforge.sources as sources
    c=Acquisition(tmp_path);calls=[]
    monkeypatch.setattr(sources.time,'sleep',lambda *_:None)
    def fail(url,**kwargs):
        calls.append(url)
        raise requests.ConnectTimeout('transient')
    monkeypatch.setattr(c.session,'get',fail)
    with pytest.raises(RuntimeError,match=r'Network failure \(ConnectTimeout\)'):
        c.get('https://www.eia.gov/x.csv')
    assert len(calls)==2
