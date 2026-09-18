# Autonomous Lead Enrichment Agent

Python pipeline that takes company domains, crawls their public sites with Playwright, converts pages to clean markdown (never raw HTML dumps), extracts structured fields with Gemini, **locks every claim to a quote that actually exists on the page**, then computes a confidence score in code.

This is the take-home for SoftwareBrio's AI Engineer Intern role.

## Why Gemini (not OpenAI / Anthropic / Groq)

The brief allows “an LLM of your choice.” Gemini 2.5 Flash is used because a **single** provider covers all three bonuses without extra keys or scraping Google SERPs:

1. Native JSON `response_schema` (Pydantic structured output).
2. **LinkedIn backfill**: Playwright searches DuckDuckGo/Bing for `{name} {company} linkedin`, then Gemini Google Search grounding if the SERP has no match (`ENABLE_LINKEDIN_SEARCH=true` by default).
3. Usage metadata for per-domain token + USD cost tracking.

The crawl loop is a custom information-gain planner (`agent/planner.py`: missing fields → score links → fetch → extract → stop), which is the spec’s “custom multi-step tool-calling loop” — not LangGraph.

## Novel piece

Typical extractors ask the model "how confident are you?" That number is cheap to fake.

This agent:

1. Scores candidate URLs by **information gain** for fields that are still missing.
2. Stops when completeness plateaus or the page/token budget is hit.
3. Requires `{value, source_url, quote}` on extracts.
4. **Drops** quotes that do not appear in the stored page text.
5. Computes `data_confidence_score` from fill rate, verified-evidence rate, source diversity, regex-confirmed emails, LinkedIn URL shape, and contradiction penalties.

## Setup

Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
playwright install chromium
copy .env.example .env
```

Set `GEMINI_API_KEY` in `.env`.

## Run (assignment test set)

```powershell
python -m lead_enrichment run --domains postman.com,supabase.com,vapi.ai --out output.json
```

Optional:

```powershell
python -m lead_enrichment run --domains postman.com --max-pages 5 --out output.json
```

## Environment

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Required |
| `GEMINI_MODEL` | Default `gemini-2.5-flash` |
| `GEMINI_INPUT_USD_PER_1M` / `GEMINI_OUTPUT_USD_PER_1M` | Cost estimate table |
| `MAX_PAGES_PER_DOMAIN` | Crawl budget |
| `MAX_CHARS_PER_PAGE` | Truncate cleaned text before the LLM |
| `PAGE_TIMEOUT_MS` | Playwright timeout |
| `ENABLE_LINKEDIN_SEARCH` | Gemini Google Search grounding for missing founder LinkedIn URLs |

Never commit `.env`. `ENABLE_LINKEDIN_SEARCH` defaults to `true` so `output.json` includes a `linkedin_search` object (`attempted`, `queries`, `filled`, `still_missing`).

## Tests

```powershell
pip install -e ".[dev]"
python -m pytest
```

Covers evidence-locking (drop quotes that are not on the page), deterministic confidence math, email sanitization, and LinkedIn hit matching.

## Output

`output.json` is a `RunReport`: one result per domain with overview, ICP, public emails, leadership (name / role / LinkedIn), evidence, pages visited, errors, token usage, estimated USD, and a confidence breakdown.

A single domain 404/timeout/bot-block does not abort the rest of the run. Status is `ok`, `partial`, or `failed`.

Resilience demo (not required for the sample file):

```powershell
python -m lead_enrichment run --domains postman.com,this-domain-should-not-resolve-xyz.test --out resilience.json
```

## Architecture

`cli` → `pipeline.enrich_domains` → per domain: Playwright fetch → HTML-to-markdown → link scoring → Gemini JSON schema extract → evidence lock → optional LinkedIn search → deterministic confidence.

## Submission note (operations question)

The role is ~40% manual lead ops and ~60% building agents that automate those ops. Submission email should include an explicit **Yes** to being comfortable with that split.
