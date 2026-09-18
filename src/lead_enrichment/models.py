from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class EvidenceSpan(BaseModel):
    source_url: str
    quote: str
    verified: bool = False


class LeadershipPerson(BaseModel):
    name: str
    role: str | None = None
    linkedin_url: str | None = None
    evidence: EvidenceSpan | None = None


class TokenUsage(BaseModel):
    model: str = ""
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            model=self.model or other.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            candidate_tokens=self.candidate_tokens + other.candidate_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=round(
                self.estimated_cost_usd + other.estimated_cost_usd, 6
            ),
        )


class ConfidenceBreakdown(BaseModel):
    fill_rate: float
    verified_evidence_rate: float
    source_diversity: float
    email_confirmation: float
    linkedin_validity: float
    contradiction_penalty: float
    score: float


class LLMEvidence(BaseModel):
    source_url: str
    quote: str


class LLMLeadership(BaseModel):
    name: str
    role: str | None = None
    linkedin_url: str | None = None
    evidence: LLMEvidence | None = None


class LLMExtraction(BaseModel):
    """Strict schema sent to Gemini. The model must not invent a confidence score."""

    company_overview: str | None = Field(
        default=None,
        description="Exactly two sentences describing what the company does.",
    )
    company_overview_evidence: LLMEvidence | None = None
    target_audience_icp: str | None = Field(
        default=None,
        description="Who the product is built for, grounded in the page text.",
    )
    icp_evidence: LLMEvidence | None = None
    contact_emails: list[str] = Field(default_factory=list)
    email_evidence: list[LLMEvidence] = Field(default_factory=list)
    leadership: list[LLMLeadership] = Field(default_factory=list)


class PageDocument(BaseModel):
    url: str
    status_code: int | None = None
    title: str = ""
    markdown: str = ""
    error: str | None = None
    blocked: bool = False


class CandidateLink(BaseModel):
    url: str
    score: float
    reason: str


class LinkedInSearchReport(BaseModel):
    attempted: bool = False
    queries: int = 0
    filled: list[str] = Field(default_factory=list)
    still_missing: list[str] = Field(default_factory=list)


class DomainResult(BaseModel):
    domain: str
    homepage_url: str
    status: Literal["ok", "partial", "failed"]
    company_overview: str | None = None
    target_audience_icp: str | None = None
    contact_points: list[str] = Field(default_factory=list)
    key_leadership: list[LeadershipPerson] = Field(default_factory=list)
    data_confidence_score: float = 0.0
    confidence_breakdown: ConfidenceBreakdown | None = None
    evidence: dict[str, EvidenceSpan | list[EvidenceSpan]] = Field(default_factory=dict)
    pages_visited: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    linkedin_search: LinkedInSearchReport = Field(default_factory=LinkedInSearchReport)


class RunReport(BaseModel):
    targets: list[str]
    results: list[DomainResult]
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class LinkedInSearchHit(BaseModel):
    name: str
    linkedin_url: HttpUrl | str
    snippet: str = ""

    @field_validator("linkedin_url", mode="before")
    @classmethod
    def coerce_url(cls, value: object) -> str:
        return str(value)
