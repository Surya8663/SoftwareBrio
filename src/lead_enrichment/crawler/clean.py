from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment
from markdownify import markdownify

DROP_TAGS = (
    "script",
    "style",
    "svg",
    "noscript",
    "iframe",
    "canvas",
    "form",
    "nav",
    "footer",
    "header",
    "aside",
)


def html_to_markdown(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda n: isinstance(n, Comment)):
        comment.extract()
    for img in soup.find_all("img"):
        img.decompose()

    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    markdown = markdownify(str(main), heading_style="ATX", strip=["img"])
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown).strip()
    if len(markdown) > max_chars:
        markdown = markdown[:max_chars].rsplit(" ", 1)[0] + "\n…[truncated]"
    return markdown
