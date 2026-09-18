from __future__ import annotations

import re
from difflib import SequenceMatcher

from lead_enrichment.models import EvidenceSpan, LLMEvidence, PageDocument

_WS = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _WS.sub(" ", value).strip().lower()


def quote_supported(quote: str, corpus: str) -> bool:
    q = normalize_text(quote)
    if len(q) < 8:
        return False
    body = normalize_text(corpus)
    if q in body:
        return True
    # Allow a short prefix/suffix miss from markdown conversion.
    if len(q) >= 24 and q[4:-4] in body:
        return True
    window = max(len(q), 24)
    step = max(12, window // 4)
    best = 0.0
    for i in range(0, max(1, len(body) - window + 1), step):
        chunk = body[i : i + window + 16]
        best = max(best, SequenceMatcher(None, q, chunk).ratio())
        if best >= 0.86:
            return True
    return False


def lock_evidence(
    evidence: LLMEvidence | None,
    pages: list[PageDocument],
) -> EvidenceSpan | None:
    if evidence is None or not evidence.quote.strip():
        return None
    by_url = {p.url: p.markdown for p in pages}
    markdown = by_url.get(evidence.source_url)
    if markdown is None:
        # Model may cite a URL that redirected; search all pages.
        for page in pages:
            if quote_supported(evidence.quote, page.markdown):
                return EvidenceSpan(
                    source_url=page.url,
                    quote=evidence.quote.strip(),
                    verified=True,
                )
        return None
    if quote_supported(evidence.quote, markdown):
        return EvidenceSpan(
            source_url=evidence.source_url,
            quote=evidence.quote.strip(),
            verified=True,
        )
    return None


def corpus_from_pages(pages: list[PageDocument]) -> str:
    return "\n\n".join(p.markdown for p in pages if p.markdown)
