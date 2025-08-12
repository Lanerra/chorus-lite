# src/chorus/models/validators.py
"""Custom validators for Pydantic models."""

from __future__ import annotations

import re

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_slug(value: str) -> str:
    """Validate that ``value`` is a slug."""
    if not SLUG_RE.match(value):
        raise ValueError("invalid slug")
    return value


def validate_non_empty(value: str) -> str:
    """Ensure ``value`` is not empty or whitespace."""
    if not value.strip():
        raise ValueError("must not be empty")
    return value


__all__ = ["validate_slug", "validate_non_empty"]
