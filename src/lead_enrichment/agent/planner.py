from __future__ import annotations

from lead_enrichment.models import CandidateLink, LeadershipPerson


def missing_fields(
    overview: str | None,
    icp: str | None,
    emails: list[str],
    leadership: list[LeadershipPerson],
) -> set[str]:
    missing: set[str] = set()
    if not overview:
        missing.add("overview")
    if not icp:
        missing.add("icp")
    if not emails:
        missing.add("contact_points")
    if not leadership:
        missing.add("leadership")
    elif any(not person.linkedin_url for person in leadership):
        missing.add("linkedin")
    return missing


def completeness(
    overview: str | None,
    icp: str | None,
    emails: list[str],
    leadership: list[LeadershipPerson],
) -> float:
    score = 0.0
    if overview:
        score += 0.25
    if icp:
        score += 0.25
    if emails:
        score += 0.25
    if leadership:
        score += 0.15
        if any(person.linkedin_url for person in leadership):
            score += 0.10
    return round(score, 3)


def should_stop(
    pages_fetched: int,
    max_pages: int,
    history: list[float],
    missing: set[str],
    pending: list[CandidateLink],
    visited: set[str],
) -> bool:
    if pages_fetched >= max_pages:
        return True
    unseen = [c for c in pending if c.url not in visited]
    if not unseen:
        return True
    crawl_missing = missing - {"linkedin"}
    if not crawl_missing:
        return True
    if len(history) >= 3 and history[-1] <= history[-3] + 0.02:
        return True
    return False


def pick_next(
    pending: list[CandidateLink],
    visited: set[str],
) -> CandidateLink | None:
    for candidate in sorted(pending, key=lambda c: c.score, reverse=True):
        if candidate.url not in visited and candidate.score >= 0.2:
            return candidate
    for candidate in pending:
        if candidate.url not in visited:
            return candidate
    return None
