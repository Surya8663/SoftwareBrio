from __future__ import annotations

import json

from google import genai
from google.genai import types

from lead_enrichment.config import Settings
from lead_enrichment.models import LLMExtraction, PageDocument, TokenUsage


EXTRACT_INSTRUCTIONS = """You extract company intelligence ONLY from the provided page text.
Rules:
- Do not invent facts, emails, people, or LinkedIn URLs that are not in the text.
- company_overview must be exactly two sentences grounded in the text.
- target_audience_icp is who the product is for, grounded in the text.
- contact_emails: only addresses that appear in the text (including mailto).
- leadership: ONLY current employees, founders, or executives OF THIS COMPANY. Do not include customers, case-study speakers, partners, or people whose title names another company.
- Every non-null field MUST include evidence.source_url matching the current page URL and a verbatim quote from the text that supports the claim.
- If a field is not supported, return null / empty lists. Prefer omission over guessing.
- Ignore cookie banners, navigation chrome, and legal boilerplate.
"""


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def extract(
        self,
        domain: str,
        page: PageDocument,
        known_summary: str,
        regex_emails: list[str],
        regex_linkedin: list[str],
    ) -> tuple[LLMExtraction, TokenUsage]:
        payload = {
            "domain": domain,
            "page_url": page.url,
            "page_title": page.title,
            "already_known": known_summary,
            "regex_emails_on_this_or_prior_pages": regex_emails,
            "regex_linkedin_urls_on_pages": regex_linkedin,
            "page_markdown": page.markdown,
        }
        last_error: Exception | None = None
        for model in _model_chain(self.settings):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=[
                        EXTRACT_INSTRUCTIONS,
                        json.dumps(payload, ensure_ascii=False),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=LLMExtraction,
                    ),
                )
                self.model = model
                usage = self._usage(response)
                parsed = _parse_extraction(response)
                return parsed, usage
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc).lower()
                if not any(token in message for token in ("429", "resource_exhausted", "quota", "unavailable", "503")):
                    raise
                continue
        raise last_error or RuntimeError("Gemini extract failed")

    def _usage(self, response: object) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
        candidates = int(getattr(meta, "candidates_token_count", 0) or 0)
        total = int(getattr(meta, "total_token_count", 0) or (prompt + candidates))
        cost = (
            prompt / 1_000_000 * self.settings.gemini_input_usd_per_1m
            + candidates / 1_000_000 * self.settings.gemini_output_usd_per_1m
        )
        return TokenUsage(
            model=self.model,
            prompt_tokens=prompt,
            candidate_tokens=candidates,
            total_tokens=total,
            estimated_cost_usd=round(cost, 6),
        )


def _model_chain(settings: Settings) -> list[str]:
    models = [settings.gemini_model]
    for item in settings.gemini_fallback_models.split(","):
        name = item.strip()
        if name and name not in models:
            models.append(name)
    return models


def _parse_extraction(response: object) -> LLMExtraction:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, LLMExtraction):
        return parsed
    text = getattr(response, "text", None) or ""
    if not text.strip():
        return LLMExtraction()
    try:
        return LLMExtraction.model_validate_json(text)
    except Exception:
        try:
            return LLMExtraction.model_validate(json.loads(text))
        except Exception:
            return LLMExtraction()
