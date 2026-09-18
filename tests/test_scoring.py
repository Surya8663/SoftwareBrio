from lead_enrichment.models import EvidenceSpan, LLMEvidence, PageDocument
from lead_enrichment.scoring.confidence import compute_confidence, result_status
from lead_enrichment.scoring.evidence import lock_evidence, quote_supported
from lead_enrichment.scoring.facts import extract_emails, valid_linkedin_url
from lead_enrichment.models import DomainResult, LeadershipPerson
from lead_enrichment.llm.search import _best_hit, _decode_bing_redirect, _unwrap
from lead_enrichment.scoring.leadership import is_internal_leader
import base64


def test_quote_supported_requires_verbatim_span() -> None:
    page = "Postman is an API platform built for engineers."
    assert quote_supported("API platform built for engineers", page)
    assert not quote_supported("this quote was never on the page at all", page)
    assert not quote_supported("short", page)


def test_lock_evidence_drops_hallucinated_quotes() -> None:
    pages = [
        PageDocument(
            url="https://example.com/about",
            markdown="Abhinav Asthana is CEO and co-founder of the company.",
        )
    ]
    good = LLMEvidence(
        source_url="https://example.com/about",
        quote="Abhinav Asthana is CEO and co-founder",
    )
    bad = LLMEvidence(
        source_url="https://example.com/about",
        quote="Jane Doe is secretly the CEO of this company",
    )
    locked = lock_evidence(good, pages)
    assert locked is not None
    assert locked.verified is True
    assert lock_evidence(bad, pages) is None
    assert lock_evidence(None, pages) is None


def test_lock_evidence_resolves_redirected_source_url() -> None:
    pages = [
        PageDocument(
            url="https://www.example.com/about",
            markdown="We help developers build backend applications quickly.",
        )
    ]
    evidence = LLMEvidence(
        source_url="https://example.com/about",
        quote="help developers build backend applications",
    )
    locked = lock_evidence(evidence, pages)
    assert locked is not None
    assert locked.source_url == "https://www.example.com/about"


def test_confidence_rewards_verified_fill_and_linkedin() -> None:
    evidence = {
        "company_overview": EvidenceSpan(
            source_url="https://a.test/", quote="we build apis", verified=True
        ),
        "target_audience_icp": EvidenceSpan(
            source_url="https://a.test/about", quote="for developers", verified=True
        ),
        "contact_points": [
            EvidenceSpan(
                source_url="https://a.test/contact",
                quote="sales@a.test",
                verified=True,
            )
        ],
    }
    people = [
        LeadershipPerson(
            name="Ada Lovelace",
            role="CEO",
            linkedin_url="https://www.linkedin.com/in/ada-lovelace",
        )
    ]
    full = compute_confidence(
        overview="We build APIs for product teams around the world today.",
        icp="Backend developers at growing software companies",
        emails=["sales@a.test"],
        leadership=people,
        evidence=evidence,
        regex_emails=["sales@a.test"],
        pages_visited=3,
    )
    empty = compute_confidence(
        overview=None,
        icp=None,
        emails=[],
        leadership=[],
        evidence={},
        regex_emails=[],
        pages_visited=0,
    )
    assert full.score > empty.score
    assert full.fill_rate == 1.0
    assert full.verified_evidence_rate == 1.0
    assert full.email_confirmation == 1.0
    assert full.linkedin_validity == 1.0
    assert empty.score == 0.0


def test_short_icp_applies_contradiction_penalty() -> None:
    breakdown = compute_confidence(
        overview="This company sells a developer platform used worldwide.",
        icp="Devs",
        emails=[],
        leadership=[],
        evidence={},
        regex_emails=[],
        pages_visited=1,
    )
    assert breakdown.contradiction_penalty == 0.08
    assert breakdown.score < 0.3 * 0.5 + 0.01


def test_result_status_partial_when_errors_and_data() -> None:
    ok = DomainResult(
        domain="a.test",
        homepage_url="https://a.test",
        status="failed",
        company_overview="Two sentences about the product live here now.",
    )
    assert result_status(ok) == "ok"
    partial = ok.model_copy(update={"errors": ["timeout"]})
    assert result_status(partial) == "partial"
    failed = DomainResult(domain="a.test", homepage_url="https://a.test", status="ok")
    assert result_status(failed) == "failed"


def test_extract_emails_strips_mailto_junk() -> None:
    text = "Reach out to [Talent@vapi.ai](mailto:Talent@vapi.ai)to speak to talent."
    assert extract_emails(text) == ["talent@vapi.ai"]
    assert "talent@vapi.ai)to" not in extract_emails(text)


def test_linkedin_url_shape() -> None:
    assert valid_linkedin_url("https://www.linkedin.com/in/paulcopplestone")
    assert not valid_linkedin_url("https://example.com/in/nope")
    assert not valid_linkedin_url(None)


def test_best_hit_matches_slug_or_full_name() -> None:
    hits = [
        {
            "name": "Paul Copplestone",
            "linkedin_url": "https://www.linkedin.com/in/paulcopplestone",
            "snippet": "CEO at Supabase",
        },
        {
            "name": "Someone Else",
            "linkedin_url": "https://www.linkedin.com/in/other-person",
            "snippet": "unrelated",
        },
    ]
    hit = _best_hit(hits, "Paul Copplestone")
    assert hit is not None
    assert "paulcopplestone" in hit["linkedin_url"]
    slug_only = _best_hit(
        [
            {
                "name": "",
                "linkedin_url": "https://www.linkedin.com/in/nathaliecriou",
                "snippet": "",
            }
        ],
        "Nathalie Criou",
    )
    assert slug_only is not None
    assert _best_hit(hits, "Totally Unknown") is None


def test_unwrap_duckduckgo_redirect() -> None:
    raw = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fabhinavasthana"
    assert "linkedin.com/in/abhinavasthana" in _unwrap(raw)


def test_decode_bing_base64_u_param() -> None:
    payload = "https://www.linkedin.com/in/paulcopplestone"
    raw = "a1" + base64.b64encode(payload.encode()).decode().rstrip("=")
    href = f"https://www.bing.com/ck/a?u={raw}"
    assert _decode_bing_redirect(href) == payload


def test_internal_leader_keeps_founders_drops_customers() -> None:
    founder = LeadershipPerson(
        name="Abhinav Asthana",
        role="CEO and co-founder",
        evidence=EvidenceSpan(
            source_url="https://www.postman.com/company/about-postman/",
            quote="Abhinav Asthana, Postman's CEO and co-founder",
            verified=True,
        ),
    )
    customer = LeadershipPerson(
        name="Seth Siegler",
        role="Chief Innovation Officer, eXp Realty",
        evidence=EvidenceSpan(
            source_url="https://supabase.com/company",
            quote="Seth Siegler, Chief Innovation Officer, eXp Realty",
            verified=True,
        ),
    )
    assert is_internal_leader(founder, "postman.com") is True
    assert is_internal_leader(customer, "supabase.com") is False
    testimonial = LeadershipPerson(
        name="Jason Mitura",
        role="VP of Software Development",
        evidence=EvidenceSpan(
            source_url="https://vapi.ai/",
            quote="Jason Mitura VP of Software Development",
            verified=True,
        ),
    )
    assert is_internal_leader(testimonial, "vapi.ai") is False
    vp = LeadershipPerson(
        name="Nathalie Criou",
        role="VP of Product",
        evidence=EvidenceSpan(
            source_url="https://vapi.ai/blog/meet-nathalie-criou",
            quote="Meet Vapi's New VP of Product",
            verified=True,
        ),
    )
    assert is_internal_leader(vp, "vapi.ai") is True
