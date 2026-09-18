from __future__ import annotations

import re

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.I,
)
MAILTO_RE = re.compile(r"mailto:([^?'\"\s>]+)", re.I)
LINKEDIN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/(?:in|company)/[A-Za-z0-9_\-%/]+",
    re.I,
)

NOISE_EMAIL_PARTS = (
    "example.com",
    "sentry.io",
    "wixpress.com",
    "cloudflare.com",
    "schema.org",
    "w3.org",
    "github.com",
    "user@email",
    "you@",
    "name@",
)


def extract_emails(text: str) -> list[str]:
    found = set(EMAIL_RE.findall(text))
    for mailto in MAILTO_RE.findall(text):
        match = EMAIL_RE.search(mailto)
        if match:
            found.add(match.group(0))
    cleaned: list[str] = []
    for raw in found:
        email = raw.strip().strip(".,);").lower()
        if not EMAIL_RE.fullmatch(email):
            continue
        if not email or any(part in email for part in NOISE_EMAIL_PARTS):
            continue
        if email.endswith((".png", ".jpg", ".svg", ".webp", ".css", ".js")):
            continue
        cleaned.append(email)
    return sorted(set(cleaned))


def extract_linkedin_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in LINKEDIN_RE.findall(text):
        url = match.split("?")[0].rstrip("/")
        if _valid_linkedin(url):
            urls.append(url)
    # de-dupe preserving order
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        key = url.lower()
        if key not in seen:
            seen.add(key)
            out.append(url)
    return out


def valid_linkedin_url(url: str | None) -> bool:
    if not url:
        return False
    return _valid_linkedin(url)


def _valid_linkedin(url: str) -> bool:
    lowered = url.lower()
    if "linkedin.com/in/" in lowered:
        slug = lowered.split("linkedin.com/in/", 1)[-1].strip("/")
        return bool(slug) and slug not in {"company", "school"}
    if "linkedin.com/company/" in lowered:
        slug = lowered.split("linkedin.com/company/", 1)[-1].strip("/")
        return bool(slug)
    return False
