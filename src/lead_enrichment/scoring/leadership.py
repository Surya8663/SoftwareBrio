from __future__ import annotations

import re

from lead_enrichment.models import LeadershipPerson

INTERNAL_TITLE = re.compile(
    r"founder|co-founder|ceo|cto|cfo|coo|chief|president|vice president|"
    r"\bvp\b|evangelist|general counsel|head of|director",
    re.I,
)
CUSTOMER_CONTEXT = re.compile(
    r"testimonial|customer stor|case study|loves (using )?|switched to",
    re.I,
)


def is_internal_leader(person: LeadershipPerson, domain: str) -> bool:
    """Keep company officers; drop customer / partner quotes that look like leadership."""
    brand = domain.split(".")[0].lower()
    role = (person.role or "").strip()
    quote = (person.evidence.quote if person.evidence else "") or ""
    blob = f"{role}\n{quote}"

    if CUSTOMER_CONTEXT.search(blob):
        return False

    if role and "," in role:
        tail = role.split(",")[-1].strip().lower()
        if tail and brand not in tail and not re.fullmatch(r"inc\.?|llc|ltd\.?", tail):
            if not INTERNAL_TITLE.search(role.split(",")[0]):
                return False
            # "Chief Innovation Officer, eXp Realty" is an external exec.
            if brand not in tail:
                return False

    source = (person.evidence.source_url if person.evidence else "") or ""
    if any(token in source.lower() for token in ("/customer", "/case-stud")):
        return brand in quote.lower() and INTERNAL_TITLE.search(quote) is not None
    if role and INTERNAL_TITLE.search(role):
        return True
    if INTERNAL_TITLE.search(quote):
        return True
    if any(token in source.lower() for token in ("/about", "/company", "/team", "/leadership")):
        return True
    return False
