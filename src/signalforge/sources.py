"""Bounded public acquisition. Retrieval is never historical publication evidence."""
import csv
from datetime import datetime
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import threading
from urllib.parse import urljoin, urlsplit, parse_qsl, urlencode, urlunsplit
import zipfile

import requests

from .runtime import atomic_json, digest, file_hash, fsync_dir, now

INDEX = {
    'tiingo': 'https://api.tiingo.com/tiingo/daily/',
    'cftc': 'https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm',
    'nport': 'https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets',
    'eia': 'https://www.eia.gov/petroleum/supply/weekly/archive/',
    'fred': 'https://api.stlouisfed.org/fred/series/observations',
    'prices': 'https://www.alphavantage.co/query',
}

SECRET_KEYS = {'api_key','apikey','token','key'}


def validate_sec_user_agent(value):
    """Require a declared contact; syntax cannot authenticate ownership."""
    if not isinstance(value,str) or '\r' in value or '\n' in value or not re.fullmatch(r'[^\r\n]+\s+[^\s@]+@[^\s@]+\.[^\s@]+',value):
        raise PermissionError('BLOCKED_AUTH: genuine SEC_USER_AGENT absent or invalid')
    domain=value.rsplit('@',1)[-1].lower()
    if domain in {'example.com','example.org','example.net'} or domain.endswith(('.invalid','.test','.example')):
        raise PermissionError('BLOCKED_AUTH: placeholder SEC contact forbidden')
    return value


def redact_metadata(value):
    if isinstance(value,dict):return {k:('REDACTED' if str(k).lower() in SECRET_KEYS else redact_metadata(v)) for k,v in value.items()}
    if isinstance(value,list):return [redact_metadata(v) for v in value]
    return value


def request_identity(url, params=None):
    # No credential bytes, including hashes of credentials, enter cache identity.
    return digest({'url':redact_url(url), 'params':{k:v for k,v in (params or {}).items()
                   if k.lower() not in SECRET_KEYS}})


def redact_url(url):
    u = urlsplit(url)
    clean = [(k, 'REDACTED' if k.lower() in SECRET_KEYS else v)
             for k, v in parse_qsl(u.query, keep_blank_values=True)]
    return urlunsplit((u.scheme, u.netloc, u.path, urlencode(clean), ''))


def allowed_url(url):
    u = urlsplit(url)
    domains = {'cftc.gov','sec.gov','eia.gov','alphavantage.co'}
    return (u.scheme == 'https' and u.port in (None, 443) and not u.username and not u.password and
            (u.hostname in {'api.tiingo.com','fred.stlouisfed.org','alfred.stlouisfed.org','api.stlouisfed.org'} or
             any(u.hostname == d or (u.hostname or '').endswith('.'+d) for d in domains)))


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.label = ''

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.href = dict(attrs).get('href')
            self.label = ''

    def handle_data(self, data):
        if self.href is not None:
            self.label += data

    def handle_endtag(self, tag):
        if tag == 'a' and self.href is not None:
            self.links.append((self.href, ' '.join(self.label.split())))
            self.href = None


def discover_zip(html, source, period, family='tff'):
    parser = Links()
    parser.feed(html)
    if source == 'cftc':
        suffix = {'tff': 'fut_fin_txt_', 'disaggregated': 'fut_disagg_txt_'}[family] + str(period) + '.zip'
        hits = {urljoin(INDEX[source], h) for h, _ in parser.links if urlsplit(h).path.lower().endswith(suffix)}
    elif source == 'nport':
        if not re.fullmatch(r'20\d\dQ[1-4]', str(period)):
            raise ValueError('Invalid quarter')
        label = str(period)[:4] + ' ' + str(period)[4:]
        hits = {urljoin(INDEX[source], h) for h, text in parser.links if text == label and h.lower().endswith('.zip')}
    else:
        raise ValueError('Unknown source')
    if len(hits) != 1 or not allowed_url(next(iter(hits))):
        raise ValueError('Missing, ambiguous, or forbidden official ZIP link')
    return hits.pop()


def discover_eia(html, page, table):
    # Match the table number in the release's HTML row; never construct a CSV URL.
    hits = set()
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', html, re.I | re.S):
        cells = re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, re.I | re.S)
        if not cells or re.sub('<[^>]+>', '', cells[0]).strip() != str(table):
            continue
        links = Links()
        links.feed(row)
        hits.update(urljoin(page, h) for h, label in links.links if 'CSV' in label.upper())
    if len(hits) != 1 or not allowed_url(next(iter(hits))):
        raise ValueError('Missing or ambiguous archived EIA CSV link')
    return hits.pop()


def validate_payload(path, kind, max_uncompressed=512*1024**2):
    with Path(path).open('rb') as stream:
        lead = stream.read(1024).lstrip().lower()
    if not lead or lead.startswith((b'<html', b'<!doctype', b'<head', b'<body')):
        raise ValueError('Empty payload or HTML denial')
    if kind in {'zip','nport_zip'}:
        limit=max_uncompressed if kind=='zip' else 8*1024**3
        with zipfile.ZipFile(path) as z:
            members = z.infolist()
            if not members or sum(m.file_size for m in members) > limit:
                raise ValueError('Empty/oversized ZIP')
            for m in members:
                p = PurePosixPath(m.filename)
                if (m.filename.startswith('/') or '\\' in m.filename or ':' in m.filename or '..' in p.parts or
                    ((m.external_attr >> 16) & 0o170000) == 0o120000 or
                    m.file_size > max(1024**2, 200*m.compress_size)):
                    raise ValueError('Unsafe ZIP path/link/compression ratio')
            if kind=='zip':
                bad = z.testzip()
                if bad:raise ValueError('ZIP CRC failure')
                verified=[m.filename for m in members]
            else:
                required={'SUBMISSION','FUND_REPORTED_INFO','REGISTRANT'};selected=[]
                for table in required:
                    hits=[m for m in members if PurePosixPath(m.filename).name.upper() in {table+'.TSV',table+'.TXT'}]
                    if len(hits)!=1:raise ValueError('Missing or ambiguous N-PORT table: '+table)
                    selected.append(hits[0])
                # Reading each selected member to EOF makes zipfile verify its CRC without inflating every holdings table.
                for member in selected:
                    with z.open(member) as stream:
                        while stream.read(1024**2):pass
                verified=[m.filename for m in selected]
        return {'members': len(members), 'uncompressed_bytes': sum(m.file_size for m in members),
                'crc_verified':kind=='zip','crc_verified_members':verified,
                'target_tables_only':kind=='nport_zip'}
    if kind == 'csv':
        if b',' not in lead and b'\t' not in lead:
            raise ValueError('Missing delimited header')
        return {'schema_qualified': False}
    if kind == 'json':
        body = json.loads(Path(path).read_text())
        if not isinstance(body, dict) or set(body) & {'error_code','Error Message','Note','Information'}:
            raise ValueError('Provider error or unexpected JSON')
        return {'schema_qualified': False}
    if kind == 'pdf':
        if not lead.startswith(b'%pdf-'):
            raise ValueError('Missing PDF signature')
        return {'structure_qualified':False,'signature_verified':True}
    if kind == 'html':
        # HTML is valid for discovery, but never validates raw numeric data.
        return {'discovery_only': True}
    raise ValueError('Unsupported payload')


class Acquisition:
    _request_lock=threading.Lock()
    _request_last=0.0
    def __init__(self, runtime, max_bytes=128*1024**2):
        if max_bytes <= 0 or max_bytes > 2*1024**3:
            raise ValueError('Acquisition budget outside pilot ceiling')
        self.root = Path(runtime) / 'raw'
        self.root.mkdir(parents=True, exist_ok=True)
        self.remaining = max_bytes
        self.session = requests.Session()
        self.last_request = 0.0

    def get(self, url, params=None):
        if params:
            url = requests.Request('GET', url, params=params).prepare().url
        for redirect in range(5):
            if not allowed_url(url):
                raise ValueError('URL outside public-source allowlist')
            host = urlsplit(url).hostname
            ua = 'SignalForge-QX/3 public research'
            headers={'User-Agent':ua, 'Accept-Encoding':'identity'}
            if host == 'api.tiingo.com':
                from .tiingo import validate_tiingo_url
                from .tiingo_actions import validate_action_url
                try:validate_tiingo_url(url)
                except PermissionError:validate_action_url(url)
                token=os.environ.get('TIINGO_API_TOKEN')
                if not token:raise PermissionError('BLOCKED_AUTH: TIINGO_API_TOKEN absent')
                headers['Authorization']='Token '+token
            if host == 'sec.gov' or host.endswith('.sec.gov'):
                ua = validate_sec_user_agent(os.environ.get('SEC_USER_AGENT', ''))
                headers['User-Agent']=ua
            attempts = 4 if host == 'api.stlouisfed.org' else 2
            for attempt in range(attempts):
                # Shared initiation limiter permits bounded I/O overlap without increasing request rate.
                with Acquisition._request_lock:
                    time.sleep(max(0, 1.0-(time.monotonic()-Acquisition._request_last)))
                    Acquisition._request_last=time.monotonic()
                    self.last_request=Acquisition._request_last
                try:
                    r = self.session.get(url, headers=headers,
                                         stream=True, allow_redirects=False, timeout=(10, 30))
                except requests.RequestException as error:
                    if attempt < attempts-1:
                        # Bounded transient retry. Never include the credential-bearing URL/message.
                        time.sleep(min(4.0, 2.0**attempt))
                        continue
                    raise RuntimeError(
                        'Network failure ('+type(error).__name__+'); credential-bearing details suppressed'
                    ) from None
                if r.status_code in (301,302,303,307,308):
                    target = urljoin(url, r.headers.get('Location',''))
                    r.close()
                    if host=='api.tiingo.com':raise ValueError('Tiingo redirects forbidden')
                    if not allowed_url(target):
                        raise ValueError('Forbidden redirect; target was not requested')
                    url = target
                    break
                if r.status_code in (403,401,429):
                    status = r.status_code
                    r.close()
                    raise PermissionError(f'BLOCKED_AUTH: HTTP {status}; no bypass or repeated retry')
                if r.status_code >= 500 and attempt < attempts-1:
                    r.close()
                    time.sleep(min(4.0, 2.0**attempt))
                    continue
                if r.status_code != 200:
                    status = r.status_code
                    detail = ''
                    if host == 'api.stlouisfed.org':
                        try:
                            chunk = next(r.iter_content(4096), b'')
                            detail = chunk.decode('utf-8','replace')
                            for secret_name in ['FRED_API_KEY','TIINGO_API_TOKEN','ALPHAVANTAGE_API_KEY']:
                                secret = os.environ.get(secret_name)
                                if secret:
                                    detail = detail.replace(secret,'[REDACTED]')
                            detail = ' '.join(detail.split())[:512]
                        except Exception:
                            detail = ''
                    r.close()
                    raise RuntimeError(f'HTTP {status}' + (f': {detail}' if detail else ''))
                return r
            else:
                raise RuntimeError('Bounded HTTP attempts exhausted')
        raise RuntimeError('Redirect bound exceeded')

    def fetch(self, url, kind, params=None, metadata=None):
        request_id = request_identity(url,params)
        folder = self.root / request_id
        folder.mkdir(exist_ok=True)
        fd, partial = tempfile.mkstemp(suffix='.partial', dir=folder)
        size = 0
        try:
            with os.fdopen(fd, 'wb') as out, self.get(url, params) as response:
                expected = response.headers.get('Content-Length')
                encoding = response.headers.get('Content-Encoding','identity')
                if expected and int(expected) > self.remaining:
                    raise ValueError('Declared response exceeds budget')
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    ceiling=2*1024**2 if kind=='tiingo' else 1024**2 if kind=='tiingo_action' else 768*1024**2 if kind=='nport_zip' else 10*1024**2 if kind=='html' else 1024**3
                    if len(chunk) > self.remaining or size > ceiling:
                        raise ValueError('Response exceeds byte budget')
                    self.remaining -= len(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
                if expected and encoding == 'identity' and size != int(expected):
                    raise ValueError('Incomplete response length')
                final_url = redact_url(response.url)
                content_type = response.headers.get('Content-Type')
            if kind == 'html':
                if b'<' not in Path(partial).read_bytes()[:1024]:
                    raise ValueError('Invalid discovery HTML')
                validation = {'discovery_only': True}
            elif kind == 'tiingo':
                from .tiingo import parse_tiingo
                values=parse_tiingo(partial,(metadata or {})['symbol'])
                validation={'schema_qualified':True,'rows':len(values),'reserved_access':False}
            elif kind == 'tiingo_action':
                from .tiingo_actions import parse_actions
                values=parse_actions(partial,(metadata or {})['symbol'],(metadata or {})['action_type'])
                validation={'schema_qualified':True,'rows':len(values),'reserved_access':False,'economic_qualified':False}
            else:
                validation = validate_payload(partial, kind)
            sha = file_hash(partial)
            dest = folder / (sha + '.' + kind)
            if dest.exists():
                if file_hash(dest) != sha:
                    raise ValueError('Existing immutable raw artifact corrupt')
                Path(partial).unlink()
            else:
                os.rename(partial, dest)
                fsync_dir(folder)
            record = {'path': str(dest), 'sha256': sha, 'bytes': size, 'source_url': final_url,
                      'retrieved_at': now(), 'content_type': content_type, 'historical_pit_certified': False,
                      'evidence_kind': 'real_pilot', 'validation': validation, 'metadata': redact_metadata(metadata or {})}
            receipt = folder / (sha + '.receipt.json')
            if not receipt.exists():
                atomic_json(receipt, record)
            return json.loads(receipt.read_text())
        except requests.RequestException:
            raise RuntimeError('Network response failure; credential-bearing details suppressed') from None
        finally:
            Path(partial).unlink(missing_ok=True)

    def html(self, url):
        receipt = self.fetch(url, 'html')
        return Path(receipt['path']).read_text(errors='replace'), receipt

    def cached(self, url, kind, params=None):
        folder=self.root/request_identity(url,params)
        candidates=sorted(folder.glob('*.receipt.json'))
        if len(candidates)>1:
            raise ValueError('Multiple raw versions require an explicit snapshot selection')
        if candidates:
            receipt=json.loads(candidates[0].read_text())
            if (not Path(receipt['path']).resolve().is_relative_to(folder.resolve()) or
                file_hash(receipt['path'])!=receipt['sha256'] or not receipt['path'].endswith('.'+kind)):
                raise ValueError('Corrupt or incompatible raw cache')
            return receipt
        return None


def parse_eia_stocks(path,release_page):
    import pandas as pd
    rows=list(csv.reader(Path(path).read_text(encoding='cp1252').splitlines()))
    if not rows or rows[0][0]!='STUB_1' or len(rows[0])<4:
        raise ValueError('Unexpected EIA stocks header')
    reference=pd.to_datetime(rows[0][1],format='%m/%d/%y')
    prior=pd.to_datetime(rows[0][2],format='%m/%d/%y')
    date=re.search(r'/([0-9]{4}_[0-9]{2}_[0-9]{2})/',release_page)
    if not date:
        raise ValueError('Archive issue date evidence absent')
    issue=pd.Timestamp(date[1].replace('_','-'))
    if not prior<reference<issue or (issue-reference).days>10:
        raise ValueError('Reference/release clock inconsistency')
    fields=['Commercial (Excluding SPR)','Total Motor Gasoline','Distillate Fuel Oil','Cushing','SPR']
    records={}
    for field in fields:
        found=[r for r in rows[1:] if r and r[0].strip()==field]
        if len(found)!=1:
            raise ValueError('EIA stocks series missing/ambiguous: '+field)
        r=found[0]
        current,previous,change=[float(v.replace(',','')) for v in r[1:4]]
        if abs((current-previous)-change)>.0021 or current<0 or previous<0:
            raise ValueError('EIA stocks difference/unit error')
        records[field]={'value':current,'prior':previous,'change':change}
    # Conservative issue-day upper bound; no nominal intraday release is invented.
    available=(issue+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
    return {'issue_date':str(issue.date()),'reference_time':reference.tz_localize('UTC').isoformat(),
            'available_at':available.isoformat(),'unit':'million_barrels','pit_tier':'B',
            'clock_evidence':release_page,'version_evidence':'dated_archive_currently_retrievable_not_independently_authenticated_original',
            'series':records}


def parse_eia_stocks_pdf(text,release_page):
    """Parse visually verified Table 4 text at its published 0.1 precision."""
    import pandas as pd
    match=re.search(r'/([0-9]{4}_[0-9]{2}_[0-9]{2})/',release_page)
    if not match or 'Million Barrels' not in text:
        raise ValueError('PDF issue/unit evidence absent')
    issue=pd.Timestamp(match[1].replace('_','-'))
    header=re.split(r'^\s*Crude Oil\s*\.',text,maxsplit=1,flags=re.M)[0]
    dates=pd.to_datetime(re.findall(r'(?<!\d)(\d{1,2}/\d{1,2}/\d{2})(?!\d)',header),format='%m/%d/%y')
    recent=sorted(set(d for d in dates if 0<(issue-d).days<=10))
    if len(recent)!=1:raise ValueError('PDF current reference missing/ambiguous')
    reference=recent[0];prior=reference-pd.Timedelta(days=7)
    if prior not in dates:raise ValueError('PDF previous-week header absent')
    fields=['Commercial (Excluding SPR)','Total Motor Gasoline','Distillate Fuel Oil','Cushing','SPR']
    series={}
    for field in fields:
        rows=[line for line in text.splitlines() if re.match(r'^\s*'+re.escape(field)+r'(?:\s|\d|\.)',line)]
        if len(rows)!=1:raise ValueError('PDF stocks row missing/ambiguous: '+field)
        tail=rows[0].split(field,1)[1]
        values=re.findall(r'(?<![\d.])-?\d[\d,]*\.\d(?!\d)',tail)
        if len(values)<3:raise ValueError('PDF published stock values missing')
        current,previous,change=[float(v.replace(',','')) for v in values[:3]]
        if min(current,previous)<0 or abs(current-previous-change)>.1501:
            raise ValueError('PDF rounded difference/unit inconsistency')
        series[field]={'value':current,'prior':previous,'change':change}
    available=(issue+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
    return {'issue_date':str(issue.date()),'reference_time':reference.tz_localize('UTC').isoformat(),
            'available_at':available.isoformat(),'unit':'million_barrels','pit_tier':'B',
            'clock_evidence':release_page,'source_format':'pdf','published_precision':.1,
            'version_evidence':'dated_archive_currently_retrievable_not_independently_authenticated_original',
            'series':series}


def parse_cftc(path, family):
    import pandas as pd
    validate_payload(path, 'zip')
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(('.txt','.csv'))]
        if len(members) != 1:
            raise ValueError('Unexpected CFTC archive schema')
        with z.open(members[0]) as f:
            frame = pd.read_csv(f, dtype=str, low_memory=False)
    frame.columns = frame.columns.str.strip()
    legacy_date='Report_Date_as_MM_DD_YYYY'
    canonical_date='Report_Date_as_YYYY-MM-DD'
    if canonical_date not in frame and legacy_date in frame:
        # Official 2010-2012 archives use this exact older schema; no fuzzy aliases.
        # Despite the old column label, the archived values are ISO dates.
        parsed=pd.to_datetime(frame[legacy_date],format='%Y-%m-%d',errors='raise')
        if 'As_of_Date_In_Form_YYMMDD' not in frame:raise ValueError('Legacy CFTC independent date field absent')
        check=pd.to_datetime(frame['As_of_Date_In_Form_YYMMDD'].str.strip().str.zfill(6),format='%y%m%d',errors='raise')
        if not parsed.equals(check):raise ValueError('Legacy CFTC date fields disagree')
        frame[canonical_date]=parsed.dt.strftime('%Y-%m-%d')
    required = {'Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','CFTC_Contract_Market_Code','Open_Interest_All'}
    if not required <= set(frame):
        raise ValueError('CFTC schema changed')
    # Official older compressed archives pad string identifiers with spaces.
    # Canonical identifiers must not change merely because the text-file layout changed.
    frame['CFTC_Contract_Market_Code']=frame['CFTC_Contract_Market_Code'].str.strip()
    frame['Market_and_Exchange_Names']=frame['Market_and_Exchange_Names'].str.strip()
    if frame['CFTC_Contract_Market_Code'].eq('').any() or frame['Market_and_Exchange_Names'].eq('').any():
        raise ValueError('Blank canonical CFTC identifier')
    frame['report_date'] = pd.to_datetime(frame['Report_Date_as_YYYY-MM-DD'], errors='raise')
    if frame.duplicated(['CFTC_Contract_Market_Code','report_date']).any():
        raise ValueError('Duplicate contract/date rows')
    frame['Open_Interest_All'] = pd.to_numeric(frame['Open_Interest_All'], errors='raise')
    if (frame['Open_Interest_All'] < 0).any():
        raise ValueError('Negative open interest')
    return frame


def parse_fred(path,frequency=None):
    import pandas as pd
    body = json.loads(Path(path).read_text())
    if 'observations' not in body:
        raise ValueError('FRED observations missing')
    records = []
    for row in body['observations']:
        reference=pd.Timestamp(row['date'])
        if frequency=='M':reference=reference+pd.offsets.MonthEnd(0)
        if pd.Timestamp(row['realtime_start'])>pd.Timestamp(row['realtime_end']):raise ValueError('Invalid vintage interval')
        value=None if row['value']=='.' else float(row['value'])
        if value is not None and not __import__('math').isfinite(value):raise ValueError('Nonfinite macro observation')
        # A date-only vintage is not an authenticated intraday release. The latest
        # end of that calendar day over global UTC offsets is next-day 12:00 UTC.
        records.append({'reference_time': reference.tz_localize('UTC').isoformat(),
                        'reference_label':row['date'],
                        'realtime_start':row['realtime_start'],
                        'available_at': (pd.Timestamp(row['realtime_start'])+pd.Timedelta(days=1,hours=12)).tz_localize('UTC').isoformat(),
                        'value': None if row['value']=='.' else float(row['value']),
                        'realtime_end': row['realtime_end'], 'pit_tier':'B',
                        'clock_evidence':'date_only_vintage_conservative_global_calendar_day_upper_bound',
                        'original_publication_qualified':False})
    # Date-only vintage bounds still require series-specific publication evidence for A.
    return records


def nport_reference_months(report_date):
    import pandas as pd
    end = pd.Timestamp(report_date).to_period('M')
    return [str(end-2), str(end-1), str(end)]
