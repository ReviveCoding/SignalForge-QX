import pytest
from signalforge.public_archive_evidence import validate_url,URLS,explain_eia_notice

def test_archive_allowlist_rejects_reserved_credentials_and_redirect_targets():
    assert validate_url(URLS['bls_cpi'])==URLS['bls_cpi']
    for url in [URLS['bls_cpi']+'?api_key=secret','https://www.bls.gov/news.release/archives/cpi_01112024.htm','https://example.com/archive','https://user:password@www.bls.gov/news.release/archives/cpi_01122023.htm']:
        with pytest.raises(PermissionError):validate_url(url)

def test_notice_explains_provider_revision_without_overriding_source_contract():
    text='December 28, 2023 Notice: correction to propane/propylene stocks introduced publication-wide revisions; weekly difference no longer aligns.'
    result=explain_eia_notice(text)
    assert result['current_protocol_arithmetic_still_fails']
    assert not result['tier_a_promotion'] and not result['registered_target_changed']
    with pytest.raises(ValueError):explain_eia_notice('unexplained difference')

def test_sec_evidence_cannot_bypass_genuine_contact_policy(monkeypatch):
    from signalforge.public_archive_evidence import fetch_public
    monkeypatch.delenv('SEC_USER_AGENT',raising=False)
    with pytest.raises(PermissionError,match='SEC_USER_AGENT'):fetch_public(URLS['sec_uso_split_2020'])

def test_large_issuer_pdf_has_separate_bounded_evidence_allowance():
    from signalforge.public_archive_evidence import byte_ceiling
    assert byte_ceiling(URLS['ishares_2023_distributions'])==8*1024**2
    assert byte_ceiling(URLS['fed_indpro'])==2*1024**2
    with pytest.raises(PermissionError):byte_ceiling('https://www.ishares.com/latest.pdf')
