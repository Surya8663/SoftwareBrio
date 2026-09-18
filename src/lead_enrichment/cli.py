from pathlib import Path

import typer

from lead_enrichment import __version__
from lead_enrichment.config import load_settings
from lead_enrichment.models import RunReport

ASSIGNMENT_DOMAINS = ("postman.com", "supabase.com", "vapi.ai")

app = typer.Typer(
    add_completion=False,
    help="Autonomous lead enrichment agent (Playwright crawl + Gemini structured extract).",
)


def _parse_domains(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("Provide at least one domain.")
    return parts


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def run(
    domains: str = typer.Option(
        ",".join(ASSIGNMENT_DOMAINS),
        help="Comma-separated company domains. Default is the assignment test set.",
    ),
    out: Path = typer.Option(Path("output.json"), help="Where to write the JSON report."),
    max_pages: int | None = typer.Option(
        None, help="Override MAX_PAGES_PER_DOMAIN for this run."
    ),
) -> None:
    """Crawl domains, extract structured intel, write output.json."""
    from lead_enrichment.pipeline import enrich_domains

    settings = load_settings()
    if not settings.gemini_api_key:
        typer.echo("GEMINI_API_KEY is missing. Copy .env.example to .env and set the key.", err=True)
        raise typer.Exit(code=2)

    target_list = _parse_domains(domains)
    if max_pages is not None:
        settings.max_pages_per_domain = max_pages

    report: RunReport = enrich_domains(target_list, settings)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out.resolve()}")
    typer.echo(
        f"Domains={len(report.results)} tokens={report.token_usage.total_tokens} "
        f"est_usd={report.token_usage.estimated_cost_usd:.6f}"
    )
    failed = [r.domain for r in report.results if r.status == "failed"]
    if failed:
        typer.echo(f"Failed domains (run continued): {', '.join(failed)}")


if __name__ == "__main__":
    app()
