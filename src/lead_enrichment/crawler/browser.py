from __future__ import annotations

import time
from urllib.parse import urljoin, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from lead_enrichment.models import PageDocument

BLOCK_HINTS = (
    "just a moment",
    "attention required",
    "verify you are human",
    "access denied",
    "captcha",
    "cloudflare",
    "enable javascript and cookies",
)


def normalize_homepage(domain: str) -> str:
    host = domain.strip().lower()
    host = host.removeprefix("https://").removeprefix("http://")
    host = host.split("/")[0]
    host = host.removeprefix("www.")
    return f"https://{host}"


def registrable_host(url_or_domain: str) -> str:
    raw = url_or_domain.strip().lower()
    if "://" not in raw:
        raw = f"https://{raw}"
    host = urlparse(raw).netloc
    return host.removeprefix("www.")


def same_site(url: str, domain: str) -> bool:
    host = registrable_host(url)
    root = registrable_host(domain)
    return host == root or host.endswith("." + root)


def absolutize(base_url: str, href: str) -> str | None:
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return None
    joined = urljoin(base_url, href)
    parsed = urlparse(joined)
    if parsed.scheme not in {"http", "https"}:
        return None
    return parsed._replace(fragment="").geturl()


class BrowserCrawler:
    def __init__(self, timeout_ms: int, retries: int) -> None:
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._pw = None
        self._browser = None
        self._context = None

    def __enter__(self) -> BrowserCrawler:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            java_script_enabled=True,
            locale="en-US",
        )
        self._context.set_default_timeout(self.timeout_ms)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def fetch_html(self, url: str) -> tuple[PageDocument, str]:
        """Return page metadata plus raw HTML. Callers must convert HTML to markdown before any LLM call."""
        last_error = "unknown fetch error"
        html = ""
        for attempt in range(self.retries + 1):
            try:
                return self._fetch_once(url)
            except PlaywrightTimeout:
                last_error = f"timeout after {self.timeout_ms}ms"
            except Exception as exc:  # noqa: BLE001 — isolate one domain from the rest
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        return PageDocument(url=url, error=last_error), html

    def _fetch_once(self, url: str) -> tuple[PageDocument, str]:
        assert self._context is not None
        page = self._context.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 12_000))
            except PlaywrightTimeout:
                pass
            status = response.status if response else None
            title = page.title() or ""
            html = page.content()
            blocked = _looks_blocked(status, title, html)
            error = None
            if status and status >= 400:
                error = f"http_{status}"
            elif blocked:
                error = "bot_block_or_challenge"
            doc = PageDocument(
                url=page.url,
                status_code=status,
                title=title,
                markdown="",
                error=error,
                blocked=blocked,
            )
            return doc, html
        finally:
            page.close()


def _looks_blocked(status: int | None, title: str, html: str) -> bool:
    if status in {401, 403, 429, 503}:
        return True
    blob = f"{title}\n{html[:4000]}".lower()
    return any(hint in blob for hint in BLOCK_HINTS)
