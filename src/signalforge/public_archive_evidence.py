"""Bounded, credential-free evidence acquisition; never a model-input adapter."""
from urllib.parse import urlsplit
import requests

URLS = {
    'eia_notice': 'https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_28/wpsr_2023_12_28.php',
    'eia_prior': 'https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_12_20/csv/table4.csv',
    'bls_cpi': 'https://www.bls.gov/news.release/archives/cpi_01122023.htm',
    'fed_indpro': 'https://www.federalreserve.gov/releases/g17/20230118/default.htm',
    'ishares_2023_distributions': 'https://www.ishares.com/us/literature/tax-information/2023-ishares-distribution-summary-stamped.pdf',
    'sec_uso_split_2020': 'https://www.sec.gov/Archives/edgar/data/1327068/000117120020000271/i20263_ex99-1.htm',
}


def validate_url(url):
    parts = urlsplit(url)
    if url not in URLS.values() or parts.scheme != 'https' or parts.query or parts.username or parts.password:
        raise PermissionError('Evidence URL outside exact development archive allowlist')
    return url


def byte_ceiling(url):
    validate_url(url)
    return 8*1024**2 if url==URLS['ishares_2023_distributions'] else 2*1024**2


def fetch_public(url, session=None):
    validate_url(url)
    if urlsplit(url).hostname=='www.sec.gov':
        # SEC acquisition must retain the repository's genuine contact policy.
        import os
        from .sources import validate_sec_user_agent
        contact=validate_sec_user_agent(os.environ.get('SEC_USER_AGENT',''))
    else:contact='SignalForge-QX/3 public archive evidence'
    session = session or requests.Session()
    # A fresh session without environment credentials or implicit HTTP authentication.
    session.trust_env = False
    session.auth = None
    session.cookies.clear()
    with session.get(url, headers={'User-Agent': contact,
                                 'Accept-Encoding': 'identity'},
                     allow_redirects=False, stream=True, timeout=(10, 30)) as response:
        if response.status_code != 200:
            raise PermissionError('Archive response HTTP '+str(response.status_code)+'; no redirect/bypass/retry')
        expected = response.headers.get('Content-Length')
        limit = byte_ceiling(url)
        if expected and int(expected) > limit: raise ValueError('Archive byte ceiling')
        chunks = []; size = 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > limit: raise ValueError('Archive byte ceiling')
            chunks.append(chunk)
        if expected and response.headers.get('Content-Encoding', 'identity') == 'identity' and size != int(expected):
            raise ValueError('Incomplete archive response')
        payload = b''.join(chunks)
        if not payload: raise ValueError('Empty archive response')
        return payload, response.headers.get('Content-Type')


def explain_eia_notice(text):
    import re
    plain = ' '.join(re.sub('<[^>]+>', ' ', text).split())
    required = ["December 28, 2023 Notice", "correction to propane/propylene stocks",
                "publication-wide revisions", "weekly difference", "no longer aligns"]
    if not all(term in plain for term in required):
        raise ValueError('Exact archived correction explanation not established')
    return {'provider_explains_mismatch': True,
            'cause': 'Prior propane/propylene correction introduced publication-wide revisions; differences use revised values while printed past levels were not republished',
            'current_protocol_arithmetic_still_fails': True,
            'tier_a_promotion': False, 'registered_target_changed': False}
