from __future__ import annotations

import json
import re

from google.genai import types

from lead_enrichment.llm.gemini_client import GeminiClient
from lead_enrichment.models import LeadershipPerson, LinkedInSearchReport, TokenUsage
from lead_enrichment.scoring.facts import LINKEDIN_RE, valid_linkedin_url

PERSON_SEARCH_PROMPT = """Search the public web for the official LinkedIn profile of this person.
Return JSON only, no markdown:
{{"name":"{name}","linkedin_url":"https://www.linkedin.com/in/...","snippet":"short reason"}}
Rules:
- linkedin_url must be a real linkedin.com/in/ (preferred) or linkedin.com/company/ URL you found.
- If you cannot find a profile, set linkedin_url to null.
Person: {name}
Role: {role}
Company domain: {domain}
"""


def fill_missing_linkedin(
    client: GeminiClient,
    domain: str,
    people: list[LeadershipPerson],
) -> tuple[list[LeadershipPerson], TokenUsage, LinkedInSearchReport]:
    missing = [p for p in people if not valid_linkedin_url(p.linkedin_url)]
    report = LinkedInSearchReport(
        attempted=bool(missing),
        queries=0,
        still_missing=[p.name for p in missing],
    )
    if not missing:
        return people, TokenUsage(model=client.model), report

    usage = TokenUsage(model=client.model)
    filled: dict[str, str] = {}
    for person in missing:
        prompt = PERSON_SEARCH_PROMPT.format(
            name=person.name,
            role=person.role or "unknown",
            domain=domain,
        )
        try:
            chat = client.client.chats.create(
                model=client.model,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            response = chat.send_message(prompt)
            usage = usage.add(client._usage(response))
            report.queries += 1
            hits = _hits_from_response(response)
            hit = _best_hit(hits, person.name)
            url = hit.get("linkedin_url") if hit else None
            if valid_linkedin_url(url):
                filled[person.name] = str(url).split("?")[0].rstrip("/")
        except Exception:
            report.queries += 1
            continue

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


def _hits_from_response(response: object) -> list[dict[str, str]]:
    text = _response_text(response)
    hits: list[dict[str, str]] = []
    json_blob = _extract_json(text)
    if json_blob:
        try:
            data = json.loads(json_blob)
            if isinstance(data, dict):
                items = data.get("hits") if isinstance(data.get("hits"), list) else [data]
            else:
                items = data if isinstance(data, list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
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
    for url in LINKEDIN_RE.findall(text):
        if valid_linkedin_url(url) and not any(h["linkedin_url"] == url for h in hits):
            hits.append({"name": "", "linkedin_url": url, "snippet": text[:280]})
    for url, title in _grounding_links(response):
        if valid_linkedin_url(url) and not any(h["linkedin_url"] == url for h in hits):
            hits.append({"name": title, "linkedin_url": url, "snippet": title})
    return hits


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def _extract_json(text: str) -> str | None:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


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
        elif len(hits) == 1:
            ranked.append((0, hit))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else None


def _grounding_links(response: object) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        meta = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(meta, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web else None
            title = getattr(web, "title", None) if web else None
            if uri:
                links.append((str(uri), str(title or "")))
    return links
