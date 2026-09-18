from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from lead_enrichment.crawler.browser import absolutize, same_site
from lead_enrichment.models import CandidateLink

PATH_SIGNALS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"/about", re.I), 1.00, "about"),
    (re.compile(r"/company", re.I), 0.95, "company"),
    (re.compile(r"/team", re.I), 0.95, "team"),
    (re.compile(r"/leadership", re.I), 0.95, "leadership"),
    (re.compile(r"/founders?", re.I), 0.92, "founders"),
    (re.compile(r"/people", re.I), 0.88, "people"),
    (re.compile(r"/contact", re.I), 0.90, "contact"),
    (re.compile(r"/pricing", re.I), 0.72, "pricing"),
    (re.compile(r"/blog/.+(meet|founder|cfo|ceo|appoint|leadership)", re.I), 0.82, "exec-blog"),
    (re.compile(r"/customers?", re.I), 0.12, "customers"),
    (re.compile(r"/product", re.I), 0.50, "product"),
    (re.compile(r"/blog", re.I), 0.25, "blog"),
    (re.compile(r"/careers|/jobs", re.I), 0.35, "careers"),
    (re.compile(r"/docs|/documentation", re.I), 0.30, "docs"),
]

CASE_STUDY = re.compile(r"/customers?/.+", re.I)
SKIP_EXT = re.compile(r"\.(pdf|png|jpe?g|gif|svg|webp|zip|mp4|css|js)$", re.I)


def discover_candidates(
    html: str,
    page_url: str,
    domain: str,
    missing_fields: set[str],
) -> list[CandidateLink]:
    soup = BeautifulSoup(html, "html.parser")
    seen: dict[str, CandidateLink] = {}
    for anchor in soup.find_all("a", href=True):
        abs_url = absolutize(page_url, anchor["href"])
        if not abs_url or not same_site(abs_url, domain):
            continue
        path = urlparse(abs_url).path or "/"
        if SKIP_EXT.search(path):
            continue
        if CASE_STUDY.search(path):
            continue
        key = _canonical(abs_url)
        score, reason = _score_path(path, missing_fields)
        if key in seen and seen[key].score >= score:
            continue
        seen[key] = CandidateLink(url=key, score=score, reason=reason)
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


def _canonical(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path, query="", fragment="").geturl()


def _score_path(path: str, missing_fields: set[str]) -> tuple[float, str]:
    base = 0.08
    reasons: list[str] = []
    for pattern, weight, label in PATH_SIGNALS:
        if pattern.search(path):
            base = max(base, weight)
            reasons.append(label)

    if "contact_points" in missing_fields and any(
        r in reasons for r in ("contact", "about", "company")
    ):
        base = min(1.0, base + 0.08)
        reasons.append("email-gap")
    if "leadership" in missing_fields and any(
        r in reasons for r in ("team", "leadership", "founders", "people", "about")
    ):
        base = min(1.0, base + 0.10)
        reasons.append("leadership-gap")
    if "overview" in missing_fields and any(
        r in reasons for r in ("about", "company", "product", "pricing")
    ):
        base = min(1.0, base + 0.05)
        reasons.append("overview-gap")

    return base, ",".join(reasons) or "generic"
