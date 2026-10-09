"""Credential-free FRED transport diagnostic. Never reads API keys."""
import json, socket, ssl
from urllib.parse import urlsplit
import requests
from signalforge.runtime import now

host='api.stlouisfed.org'
out={'observed_at':now(),'host':host,'credential_read':False}
try:
    out['dns']=sorted({x[4][0] for x in socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)})
except Exception as e:
    out['dns_error_type']=type(e).__name__

try:
    context=ssl.create_default_context()
    with socket.create_connection((host,443),timeout=10) as sock:
        with context.wrap_socket(sock,server_hostname=host) as tls:
            out['tls_version']=tls.version()
            cert=tls.getpeercert()
            out['tls_subject']=dict(x[0] for x in cert.get('subject',[])).get('commonName')
except Exception as e:
    out['tls_error_type']=type(e).__name__

try:
    r=requests.get(
        'https://api.stlouisfed.org/fred/series/observations',
        params={'series_id':'UNRATE','file_type':'json'},
        timeout=(10,30),allow_redirects=False,
        headers={'User-Agent':'SignalForge-QX/3 transport diagnostic','Accept-Encoding':'identity'},
    )
    out['http_status']=r.status_code
    out['content_type']=r.headers.get('Content-Type')
    out['response_bytes']=len(r.content)
    try:
        body=r.json()
        out['provider_error_code']=body.get('error_code')
        out['provider_error_message']=body.get('error_message')
    except Exception:
        out['json_body']=False
except requests.RequestException as e:
    out['requests_error_type']=type(e).__name__
    if getattr(e,'__cause__',None) is not None:
        out['cause_type']=type(e.__cause__).__name__

print(json.dumps(out,indent=2))
