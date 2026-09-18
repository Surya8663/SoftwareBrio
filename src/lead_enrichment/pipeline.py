from __future__ import annotations

from lead_enrichment.agent.planner import completeness, missing_fields, pick_next, should_stop
from lead_enrichment.config import Settings
from lead_enrichment.crawler.browser import BrowserCrawler, normalize_homepage
from lead_enrichment.crawler.clean import html_to_markdown
from lead_enrichment.crawler.discover import discover_candidates
from lead_enrichment.llm.gemini_client import GeminiClient
from lead_enrichment.llm.search import fill_missing_linkedin
from lead_enrichment.models import (
    CandidateLink,
    DomainResult,
    EvidenceSpan,
    LeadershipPerson,
    PageDocument,
    RunReport,
    TokenUsage,
)
from lead_enrichment.scoring.confidence import compute_confidence, result_status
from lead_enrichment.scoring.evidence import lock_evidence
from lead_enrichment.scoring.facts import extract_emails, extract_linkedin_urls, valid_linkedin_url


def enrich_domains(domains: list[str], settings: Settings) -> RunReport:
    totals = TokenUsage(model=settings.gemini_model)
    results: list[DomainResult] = []
    llm = GeminiClient(settings)
    with BrowserCrawler(settings.page_timeout_ms, settings.request_retries) as crawler:
        for domain in domains:
            try:
                result = enrich_one(domain, settings, crawler, llm)
            except Exception as exc:  # noqa: BLE001 — never abort the whole run
                result = DomainResult(
                    domain=domain,
                    homepage_url=normalize_homepage(domain),
                    status="failed",
                    errors=[f"unhandled: {type(exc).__name__}: {exc}"],
                )
            results.append(result)
            totals = totals.add(result.token_usage)
    return RunReport(targets=list(domains), results=results, token_usage=totals)


def enrich_one(
    domain: str,
    settings: Settings,
    crawler: BrowserCrawler,
    llm: GeminiClient,
) -> DomainResult:
    homepage = normalize_homepage(domain)
    pages: list[PageDocument] = []
    visited: set[str] = set()
    pending: list[CandidateLink] = []
    errors: list[str] = []
    usage = TokenUsage(model=settings.gemini_model)
    history: list[float] = []

    overview: str | None = None
    overview_ev: EvidenceSpan | None = None
    icp: str | None = None
    icp_ev: EvidenceSpan | None = None
    emails: list[str] = []
    email_ev: list[EvidenceSpan] = []
    leadership: list[LeadershipPerson] = []
    regex_emails: list[str] = []
    regex_linkedin: list[str] = []

    def snapshot() -> float:
        return completeness(overview, icp, emails, leadership)

    seed = CandidateLink(url=homepage, score=1.0, reason="homepage")
    pending.append(seed)

    while not should_stop(
        len(pages),
        settings.max_pages_per_domain,
        history,
        missing_fields(overview, icp, emails, leadership),
        pending,
        visited,
    ):
        nxt = pick_next(pending, visited)
        if nxt is None:
            break
        visited.add(nxt.url)
        pending = [c for c in pending if c.url not in visited]
        doc, html = crawler.fetch_html(nxt.url)
        if doc.error:
            errors.append(f"{nxt.url}: {doc.error}")
            if doc.blocked or not html:
                history.append(snapshot())
                continue
        markdown = html_to_markdown(html, settings.max_chars_per_page) if html else ""
        if not markdown.strip():
            errors.append(f"{doc.url}: empty_content")
            history.append(snapshot())
            continue
        doc.markdown = markdown
        pages.append(doc)

        regex_emails = sorted(set(regex_emails) | set(extract_emails(markdown)))
        regex_linkedin = _merge_urls(regex_linkedin, extract_linkedin_urls(markdown))
        for extra in discover_candidates(
            html, doc.url, domain, missing_fields(overview, icp, emails, leadership)
        ):
            if extra.url not in visited and extra.url not in {c.url for c in pending}:
                pending.append(extra)

        known = _known_summary(overview, icp, emails, leadership)
        try:
            extraction, step_usage = llm.extract(
                domain, doc, known, regex_emails, regex_linkedin
            )
            usage = usage.add(step_usage)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"llm_extract {doc.url}: {type(exc).__name__}: {exc}")
            history.append(snapshot())
            continue

        ov_ev = lock_evidence(extraction.company_overview_evidence, pages)
        if extraction.company_overview and ov_ev and not overview:
            overview = extraction.company_overview.strip()
            overview_ev = ov_ev

        icp_locked = lock_evidence(extraction.icp_evidence, pages)
        if extraction.target_audience_icp and icp_locked and not icp:
            icp = extraction.target_audience_icp.strip()
            icp_ev = icp_locked

        for proposed in extraction.contact_emails:
            addr = proposed.strip().lower()
            if addr in emails:
                continue
            if addr in {e.lower() for e in regex_emails}:
                emails.append(addr)
            else:
                # Accept only if a locked quote on some page contains the address.
                supported = False
                for ev in extraction.email_evidence:
                    locked = lock_evidence(ev, pages)
                    if locked and addr in locked.quote.lower():
                        emails.append(addr)
                        email_ev.append(locked)
                        supported = True
                        break
                if not supported:
                    continue
        for addr in regex_emails:
            if addr not in emails:
                emails.append(addr)
            if not any(
                isinstance(ev, EvidenceSpan) and addr in ev.quote.lower()
                for ev in email_ev
            ):
                span = _email_span(addr, pages)
                if span:
                    email_ev.append(span)

        for person in extraction.leadership:
            if not person.name or _has_person(leadership, person.name):
                continue
            ev = lock_evidence(person.evidence, pages)
            if ev is None:
                continue
            linkedin = person.linkedin_url if valid_linkedin_url(person.linkedin_url) else None
            if linkedin and linkedin not in regex_linkedin:
                # URL must appear in crawled text to count as on-site.
                if not any(linkedin.lower() in p.markdown.lower() for p in pages):
                    linkedin = None
            if not linkedin:
                linkedin = _linkedin_near_name(person.name, regex_linkedin, pages)
            leadership.append(
                LeadershipPerson(
                    name=person.name.strip(),
                    role=(person.role or None),
                    linkedin_url=linkedin,
                    evidence=ev,
                )
            )

        history.append(snapshot())

    if (
        settings.enable_linkedin_search
        and leadership
        and any(not valid_linkedin_url(p.linkedin_url) for p in leadership)
    ):
        try:
            leadership, search_usage = fill_missing_linkedin(llm, domain, leadership)
            usage = usage.add(search_usage)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"linkedin_search: {type(exc).__name__}: {exc}")

    evidence: dict[str, EvidenceSpan | list[EvidenceSpan]] = {}
    if overview_ev:
        evidence["company_overview"] = overview_ev
    if icp_ev:
        evidence["target_audience_icp"] = icp_ev
    if email_ev:
        evidence["contact_points"] = email_ev
    leader_ev = [p.evidence for p in leadership if p.evidence]
    if leader_ev:
        evidence["key_leadership"] = leader_ev

    breakdown = compute_confidence(
        overview, icp, emails, leadership, evidence, regex_emails, len(pages)
    )
    result = DomainResult(
        domain=domain,
        homepage_url=homepage,
        status="failed",
        company_overview=overview,
        target_audience_icp=icp,
        contact_points=emails,
        key_leadership=leadership,
        data_confidence_score=breakdown.score,
        confidence_breakdown=breakdown,
        evidence=evidence,
        pages_visited=[p.url for p in pages],
        errors=errors,
        token_usage=usage,
    )
    result.status = result_status(result)  # type: ignore[assignment]
    if not pages and errors:
        result.status = "failed"
    return result


def _known_summary(
    overview: str | None,
    icp: str | None,
    emails: list[str],
    leadership: list[LeadershipPerson],
) -> str:
    return (
        f"overview={overview or ''}\n"
        f"icp={icp or ''}\n"
        f"emails={', '.join(emails)}\n"
        f"leadership={', '.join(p.name for p in leadership)}"
    )


def _has_person(people: list[LeadershipPerson], name: str) -> bool:
    target = name.strip().lower()
    return any(p.name.strip().lower() == target for p in people)


def _merge_urls(existing: list[str], new: list[str]) -> list[str]:
    out = list(existing)
    seen = {u.lower() for u in existing}
    for url in new:
        if url.lower() not in seen:
            seen.add(url.lower())
            out.append(url)
    return out


def _email_span(addr: str, pages: list[PageDocument]) -> EvidenceSpan | None:
    needle = addr.lower()
    for page in pages:
        idx = page.markdown.lower().find(needle)
        if idx < 0:
            continue
        start = max(0, idx - 80)
        end = min(len(page.markdown), idx + len(addr) + 80)
        quote = page.markdown[start:end].strip()
        return EvidenceSpan(source_url=page.url, quote=quote, verified=True)
    return None


def _linkedin_near_name(
    name: str, urls: list[str], pages: list[PageDocument]
) -> str | None:
    token = name.split()[0].lower() if name.split() else ""
    if not token:
        return None
    for url in urls:
        if token in url.lower():
            return url
    for page in pages:
        idx = page.markdown.lower().find(name.lower())
        if idx < 0:
            continue
        window = page.markdown[max(0, idx - 400) : idx + 400]
        for url in urls:
            if url in window:
                return url
    return None
