from __future__ import annotations

import base64
import html as html_lib
import re
import time
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from lead_enrichment.crawler.browser import BrowserCrawler
from lead_enrichment.crawler.clean import html_to_markdown
from lead_enrichment.models import LeadershipPerson, LinkedInSearchReport, TokenUsage
from lead_enrichment.scoring.facts import LINKEDIN_RE, extract_linkedin_urls, valid_linkedin_url


def fill_missing_linkedin(
    domain: str,
    people: list[LeadershipPerson],
    crawler: BrowserCrawler,
    llm: object | None = None,
) -> tuple[list[LeadershipPerson], TokenUsage, LinkedInSearchReport]:
    """Fill missing LinkedIn URLs. Browser search is primary; Gemini grounding is backup."""
    missing = [p for p in people if not valid_linkedin_url(p.linkedin_url)]
    report = LinkedInSearchReport(
        attempted=bool(missing),
        queries=0,
        still_missing=[p.name for p in missing],
    )
    usage = TokenUsage(model=getattr(llm, "model", "") if llm else "browser-search")
    if not missing:
        return people, usage, report

    filled: dict[str, str] = {}
    for person in missing:
        report.queries += 1
        url = search_profile_url(crawler, person.name, domain, person.role)
        if valid_linkedin_url(url):
            filled[person.name] = str(url).split("?")[0].rstrip("/")
            time.sleep(1.2)
            continue
        if llm is not None:
            gemini_url, step = _gemini_one(llm, domain, person)
            usage = usage.add(step)
            if valid_linkedin_url(gemini_url):
                filled[person.name] = str(gemini_url).split("?")[0].rstrip("/")

    updated: list[LeadershipPerson] = []
    still: list[str] = []
    for person in people:
        if valid_linkedin_url(person.linkedin_url):
            updated.append(person)
            continue
        url = filled.get(person.name)
        if valid_linkedin_url(url):
            report.filled.append(person.name)
            updated.append(person.model_copy(update={"linkedin_url": url}))
        else:
            still.append(person.name)
            updated.append(person)
    report.still_missing = still
    return updated, usage, report


def search_profile_url(
    crawler: BrowserCrawler,
    name: str,
    domain: str,
    role: str | None = None,
) -> str | None:
    brand = domain.split(".")[0]
    query = " ".join(part for part in (name, brand, "linkedin") if part)
    bing = f"https://www.bing.com/search?q={quote_plus(query)}"
    doc, html = crawler.fetch_html(bing, settle=False)
    if html:
        for target in _bing_result_targets(html, name):
            if valid_linkedin_url(target):
                return target.split("?")[0].rstrip("/")
            followed = _follow(crawler, target)
            if followed:
                return followed
        hits = _hits_from_html(html)
        best = _best_hit(hits, name)
        if best:
            return best["linkedin_url"]

    ddg = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    _doc, ddg_html = crawler.fetch_html(ddg, settle=False)
    if ddg_html:
        hits = _hits_from_html(ddg_html)
        best = _best_hit(hits, name)
        if best:
            return best["linkedin_url"]
    return None


def _bing_result_targets(html: str, name: str) -> list[str]:
    parts = [p.lower() for p in re.split(r"\s+", name.strip()) if len(p) > 1]
    last = parts[-1] if parts else ""
    targets: list[str] = []
    seen: set[str] = set()
    for href, title_html in re.findall(
        r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.I | re.S,
    ):
        title = re.sub(r"<[^>]+>", " ", html_lib.unescape(title_html))
        title = re.sub(r"\s+", " ", title).strip().lower()
        url = html_lib.unescape(href)
        if url.startswith("/"):
            url = "https://www.bing.com" + url
        decoded = _decode_bing_redirect(url)
        dest = decoded or url
        name_in_title = bool(parts) and all(part in title for part in parts)
        last_in_dest = bool(last) and last in dest.lower()
        if valid_linkedin_url(dest) and (name_in_title or last_in_dest):
            key = dest.split("?")[0].rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                targets.append(dest.split("?")[0].rstrip("/"))
            continue
        if "linkedin" in title and name_in_title:
            if url not in seen:
                seen.add(url)
                targets.append(url)
    return targets


def _decode_bing_redirect(url: str) -> str | None:
    parsed = urlparse(url.replace("&amp;", "&"))
    raw = (parse_qs(parsed.query).get("u") or [None])[0]
    if not raw:
        return None
    raw = unquote(raw)
    if raw.startswith("a1"):
        raw = raw[2:]
    pad = "=" * (-len(raw) % 4)
    try:
        decoded = base64.b64decode(raw + pad).decode("utf-8", "ignore")
    except Exception:
        return None
    return decoded if decoded.startswith("http") else None


def _follow(crawler: BrowserCrawler, url: str) -> str | None:
    doc, _html = crawler.fetch_html(url, settle=False)
    final = doc.url.split("?")[0].rstrip("/")
    if valid_linkedin_url(final):
        return final
    return None


def _hits_from_html(html: str) -> list[dict[str, str]]:
    text = html_to_markdown(html, 20_000)
    urls = extract_linkedin_urls(html) + extract_linkedin_urls(text)
    for raw in re.findall(r"https?://[^\s\"'<>]+", html):
        unwrapped = _unwrap(raw)
        if valid_linkedin_url(unwrapped):
            urls.append(unwrapped.split("?")[0].rstrip("/"))
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for url in urls:
        key = url.lower()
        if key in seen or not valid_linkedin_url(url):
            continue
        seen.add(key)
        hits.append({"name": "", "linkedin_url": url, "snippet": text[:400]})
    return hits


def _unwrap(url: str) -> str:
    parsed = urlparse(url.replace("&amp;", "&"))
    qs = parse_qs(parsed.query)
    for key in ("uddg", "u", "url"):
        if key in qs:
            return unquote(qs[key][0])
    return url


def _best_hit(hits: list[dict[str, str]], name: str) -> dict[str, str] | None:
    parts = [p.lower() for p in re.split(r"\s+", name.strip()) if len(p) > 1]
    if not parts:
        return None
    last = parts[-1]
    ranked: list[tuple[int, dict[str, str]]] = []
    for hit in hits:
        if not valid_linkedin_url(hit.get("linkedin_url")):
            continue
        blob = f"{hit.get('name', '')} {hit.get('snippet', '')} {hit.get('linkedin_url', '')}".lower()
        if all(part in blob for part in parts):
            ranked.append((2, hit))
        elif len(last) >= 4 and last in (hit.get("linkedin_url") or "").lower():
            ranked.append((1, hit))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else None


def _gemini_one(llm: object, domain: str, person: LeadershipPerson) -> tuple[str | None, TokenUsage]:
    from google.genai import types

    prompt = (
        "Find the official LinkedIn profile URL. Return JSON only: "
        f'{{"linkedin_url":"https://www.linkedin.com/in/..."}}\n'
        f"Person: {person.name}\nRole: {person.role or 'unknown'}\nCompany: {domain}"
    )
    try:
        chat = llm.client.chats.create(  # type: ignore[attr-defined]
            model=llm.model,  # type: ignore[attr-defined]
            config=types.GenerateContentConfig(
                temperature=0.1,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        response = chat.send_message(prompt)
        usage = llm._usage(response)  # type: ignore[attr-defined]
        text = getattr(response, "text", "") or ""
        match = LINKEDIN_RE.search(text)
        return (match.group(0) if match else None), usage
    except Exception:
        return None, TokenUsage(model=getattr(llm, "model", ""))
