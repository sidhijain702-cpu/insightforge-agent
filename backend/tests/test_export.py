import jwt

from app import export
from app.config import settings

MD = "# T\n\n## S\n\n**Confidence: HIGH**\n\nFact. [Source: a.com, 2026-01-01, credibility: 0.80](cite:E1) [UNVERIFIED]\n"


def test_markdown_export_drops_internal_cite_targets():
    out = export.export_markdown(MD)
    assert "(cite:" not in out and "[Source: a.com, 2026-01-01, credibility: 0.80]" in out


def test_share_token_roundtrip_and_tamper_detection():
    token, exp = export.make_share_token("rpt_123")
    assert export.verify_share_token(token) == "rpt_123"
    try:
        export.verify_share_token(token[:-3] + "abc")
        raise AssertionError("tampered token accepted")
    except jwt.PyJWTError:
        pass


def test_expired_token_rejected():
    old = settings.share_ttl_days
    settings.share_ttl_days = -1
    try:
        token, _ = export.make_share_token("rpt_1")
    finally:
        settings.share_ttl_days = old
    try:
        export.verify_share_token(token)
        raise AssertionError("expired token accepted")
    except jwt.ExpiredSignatureError:
        pass


def test_html_has_badges_and_marks():
    try:
        html = export.to_html(MD)
    except ImportError:   # markdown lib missing in minimal environments
        return
    assert "badge high" in html and "<mark>" in html and "cite:" not in html


def test_pdf_fallback_text_keeps_typographic_punctuation_readable():
    out = export._to_latin1("The region\u2019s real\u2011time \u201cplan\u201d \u2013 cost \u20ac5 \u2026")
    assert out == 'The region\'s real-time "plan" - cost EUR 5 ...'
    assert "?" not in out
