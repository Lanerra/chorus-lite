# src/chorus/core/name_utils.py
"""Name normalization utilities for Chorus.

All internal processing should use slugified identifiers. Any user-facing
display should render a human-friendly display name derived from the slug.

Rules:
- slugify_name: lowercase, ASCII fold, keep [a-z0-9-], collapse hyphens
- display_from_slug: split hyphens, title-case tokens with exceptions map
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


def slugify_name(name: str) -> str:
    """Return a canonical slug identifier from free-form ``name``.

    This function performs:
    - Unicode NFKD normalization and ASCII folding
    - Lowercasing
    - Whitespace to single hyphen conversion
    - Removal of any character not [a-z0-9-]
    - Collapse of multiple hyphens
    - Trimming of leading/trailing hyphens

    Examples:
        - "Neo Argent" -> "neo-argent"
        - "Tylia—Verdant" -> "tylia-verdant"
        - "Iron Vassals: Kael" -> "iron-vassals-kael"
    """
    # Normalize unicode to compatibility decomposition (NFKD)
    try:
        name = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    except Exception:
        name = str(name)

    # Lowercase and normalize whitespace
    s = re.sub(r"\s+", "-", name.strip().lower())
    # Remove any non [a-z0-9-]
    s = re.sub(r"[^a-z0-9-]", "", s)
    # Collapse repeated hyphens and trim
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


# Small words that should remain lowercase when not first/last
_SMALL_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "in",
    "on",
    "to",
    "for",
    "at",
    "by",
    "from",
    "with",
    "as",
}

# Known acronyms/initialisms to render verbatim (uppercase)
_ACRONYMS = {
    "ai",
    "ii",
    "iii",
    "iv",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}

# Tokens that should be cased as-is (first letter uppercase, rest lowercase)
_PROPER_TOKENS = {
    # Extend as needed for world-specific conventions
    "neo",
}


def _title_token(tok: str, index: int, last_index: int) -> str:
    # Acronyms and roman numerals
    if tok in _ACRONYMS:
        return tok.upper()

    # Small words lowercased unless first or last token
    if 0 < index < last_index and tok in _SMALL_WORDS:
        return tok

    # Proper tokens (domain exceptions) title-cased
    if tok in _PROPER_TOKENS:
        return tok.capitalize()

    # Default: Title-case
    return tok.capitalize()


def display_from_slug(slug: str) -> str:
    """Return a human-friendly display name from a canonical ``slug``.

    Examples:
        - "neo-argent" -> "Neo Argent"
        - "tylia-verdant" -> "Tylia Verdant"
        - "iron-vassals-kael" -> "Iron Vassals Kael"
    """
    # If input is not already a slug, coerce through slugify_name for safety.
    s = slugify_name(slug)
    if not s:
        return ""

    parts = [p for p in s.split("-") if p]
    if not parts:
        return ""

    last = len(parts) - 1
    return " ".join(_title_token(tok, i, last) for i, tok in enumerate(parts))


def bulk_slugify(names: Iterable[str]) -> list[str]:
    """Slugify a sequence of names with dedupe preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        s = slugify_name(n)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
