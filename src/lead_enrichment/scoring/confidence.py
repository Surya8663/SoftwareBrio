from __future__ import annotations

from lead_enrichment.models import (
    ConfidenceBreakdown,
    DomainResult,
    EvidenceSpan,
    LeadershipPerson,
)
from lead_enrichment.scoring.facts import valid_linkedin_url


def compute_confidence(
    overview: str | None,
    icp: str | None,
    emails: list[str],
    leadership: list[LeadershipPerson],
    evidence: dict[str, EvidenceSpan | list[EvidenceSpan]],
    regex_emails: list[str],
    pages_visited: int,
) -> ConfidenceBreakdown:
    filled = 0
    filled += 1 if overview else 0
    filled += 1 if icp else 0
    filled += 1 if emails else 0
    filled += 1 if leadership else 0
    fill_rate = filled / 4.0

    claims: list[EvidenceSpan] = []
    for value in evidence.values():
        if isinstance(value, list):
            claims.extend(value)
        elif isinstance(value, EvidenceSpan):
            claims.append(value)
    if claims:
        verified_rate = sum(1 for c in claims if c.verified) / len(claims)
    else:
        verified_rate = 0.0

    sources = {c.source_url for c in claims if c.verified}
    source_diversity = min(1.0, len(sources) / 3.0) if pages_visited else 0.0

    if emails and set(e.lower() for e in emails) & set(e.lower() for e in regex_emails):
        email_confirmation = 1.0
    elif emails and verified_rate >= 0.5:
        email_confirmation = 0.55
    elif emails:
        email_confirmation = 0.2
    else:
        email_confirmation = 0.0

    if not leadership:
        linkedin_validity = 0.0
    else:
        linkedin_validity = sum(
            1 for person in leadership if valid_linkedin_url(person.linkedin_url)
        ) / len(leadership)

    contradiction_penalty = _contradiction_penalty(overview, icp)

    score = (
        0.30 * fill_rate
        + 0.30 * verified_rate
        + 0.15 * source_diversity
        + 0.15 * email_confirmation
        + 0.10 * linkedin_validity
        - contradiction_penalty
    )
    score = max(0.0, min(1.0, round(score, 3)))
    return ConfidenceBreakdown(
        fill_rate=round(fill_rate, 3),
        verified_evidence_rate=round(verified_rate, 3),
        source_diversity=round(source_diversity, 3),
        email_confirmation=round(email_confirmation, 3),
        linkedin_validity=round(linkedin_validity, 3),
        contradiction_penalty=round(contradiction_penalty, 3),
        score=score,
    )


def result_status(result: DomainResult) -> str:
    if result.company_overview or result.contact_points or result.key_leadership:
        if result.errors:
            return "partial"
        return "ok"
    return "failed"


def _contradiction_penalty(overview: str | None, icp: str | None) -> float:
    if not overview or not icp:
        return 0.0
    # Extremely short leftover after lock often means the model hallucinated then we stripped it.
    if len(overview.split()) < 8 or len(icp.split()) < 4:
        return 0.08
    return 0.0
