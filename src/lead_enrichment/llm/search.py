from __future__ import annotations

import json
import re

from google.genai import types

from lead_enrichment.llm.gemini_client import GeminiClient
from lead_enrichment.models import LeadershipPerson, TokenUsage
from lead_enrichment.scoring.facts import LINKEDIN_RE, valid_linkedin_url

SEARCH_PROMPT = """Find official LinkedIn profile URLs for these people at this company.
Return JSON only:
{{"hits":[{{"name":"","linkedin_url":"https://www.linkedin.com/in/...","snippet":""}}]}}
Only include linkedin.com/in/ or linkedin.com/company/ URLs you actually found.
People:
{people}
Company domain: {domain}
"""


def fill_missing_linkedin(
    client: GeminiClient,
    domain: str,
    people: list[LeadershipPerson],
) -> tuple[list[LeadershipPerson], TokenUsage]:
    missing = [p for p in people if not valid_linkedin_url(p.linkedin_url)]
    if not missing:
        return people, TokenUsage(model=client.model)

    listing = "\n".join(
        f"- {p.name}" + (f" ({p.role})" if p.role else "") for p in missing
    )
    response = client.client.models.generate_content(
        model=client.model,
        contents=SEARCH_PROMPT.format(people=listing, domain=domain),
        config=types.GenerateContentConfig(
            temperature=0.1,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    usage = client._usage(response)
    hits = _hits_from_response(response)
    by_name = {h["name"].strip().lower(): h for h in hits if h.get("name")}
    updated: list[LeadershipPerson] = []
    for person in people:
        if valid_linkedin_url(person.linkedin_url):
            updated.append(person)
            continue
        hit = by_name.get(person.name.strip().lower())
        url = hit.get("linkedin_url") if hit else None
        if not url:
            # Fall back to any grounded URL whose snippet mentions the person.
            url = _url_mentioning(hits, person.name)
        if valid_linkedin_url(url):
            snippet = (hit or {}).get("snippet") or url
            updated.append(
                person.model_copy(
                    update={
                        "linkedin_url": url.split("?")[0].rstrip("/"),
                        "evidence": person.evidence,
                    }
                )
            )
            # Keep evidence from the site if present; search snippet is not site-locked.
            _ = snippet
        else:
            updated.append(person)
    return updated, usage


def _hits_from_response(response: object) -> list[dict[str, str]]:
    text = getattr(response, "text", None) or ""
    hits: list[dict[str, str]] = []
    json_blob = _extract_json(text)
    if json_blob:
        try:
            data = json.loads(json_blob)
            for item in data.get("hits", []):
                url = str(item.get("linkedin_url") or "")
                if valid_linkedin_url(url):
                    hits.append(
                        {
                            "name": str(item.get("name") or ""),
                            "linkedin_url": url,
                            "snippet": str(item.get("snippet") or ""),
                        }
                    )
        except json.JSONDecodeError:
            pass
    if not hits:
        for url in LINKEDIN_RE.findall(text):
            if valid_linkedin_url(url):
                hits.append({"name": "", "linkedin_url": url, "snippet": text[:280]})
    grounding = _grounding_urls(response)
    for url in grounding:
        if valid_linkedin_url(url) and not any(h["linkedin_url"] == url for h in hits):
            hits.append({"name": "", "linkedin_url": url, "snippet": "grounding chunk"})
    return hits


def _extract_json(text: str) -> str | None:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def _url_mentioning(hits: list[dict[str, str]], name: str) -> str | None:
    token = name.split()[0].lower() if name.split() else ""
    for hit in hits:
        blob = f"{hit.get('name','')} {hit.get('snippet','')} {hit.get('linkedin_url','')}".lower()
        if token and token in blob and valid_linkedin_url(hit.get("linkedin_url")):
            return hit["linkedin_url"]
    for hit in hits:
        if valid_linkedin_url(hit.get("linkedin_url")):
            return hit["linkedin_url"]
    return None


def _grounding_urls(response: object) -> list[str]:
    urls: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        meta = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(meta, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web else None
            if uri:
                urls.append(str(uri))
    return urls
