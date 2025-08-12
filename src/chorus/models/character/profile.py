# src/chorus/models/character/profile.py
"""Data model for a character profile.

Adds split models for update vs. generation with minimal contracts,
while keeping the existing CharacterProfile model intact for canonical storage.
"""

from __future__ import annotations

from datetime import date

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from ..mixins import IDMixin, TimestampsMixin
from ..validators import (
    validate_non_empty,
)


class CharacterProfile(IDMixin):
    """Character profile for a character."""

    id: str | None = Field(default=None)
    name: str = Field(..., min_length=1)
    full_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    gender: str | None = None
    age: int | None = None
    species: str | None = None
    role: str | None = None

    description: str = ""
    status: str = "Unknown"
    backstory: str | None = None
    fatal_flaw: str | None = None
    development_history: dict[str, list[str]] | None = None
    created_chapter: int = 0
    version: int = 1

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        # Allow human-readable names; only require non-empty.
        return validate_non_empty(v)

    @field_validator("development_history", mode="before")
    @classmethod
    def _validate_development_history(cls, v: dict[str, list[str]] | None) -> dict[str, list[str]] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("development_history must be a dictionary")
        # Validate each chapter key and its list of events
        for chapter, events in v.items():
            if not isinstance(chapter, str) or not chapter:
                raise ValueError("Chapter keys must be non-empty strings")
            if not isinstance(events, list):
                raise ValueError(f"Events for chapter {chapter} must be a list")
            # Validate each event
            for event in events:
                if not isinstance(event, str) or not event.strip():
                    raise ValueError(f"Event '{event}' is invalid")
        return v

    @field_validator("aliases", mode="before")
    @classmethod
    def _validate_lists(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("must be a list")
        return [item.strip() for item in v if item.strip()]


class CharacterProfileUpdate:
    """Update payload for an existing character profile (minimal contract)."""

    model_config = ConfigDict(extra="forbid")

    id: str  # UUID acceptable; keep as string for minimal contract
    name: str | None = None
    description: str | None = None
    status: str | None = None
    role: str | None = None


class CharacterProfileGenerate:
    """Generation payload for creating a new character profile (minimal contract)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str | None = None
    status: str | None = None
    role: str | None = None


__all__ = ["CharacterProfile", "CharacterProfileUpdate", "CharacterProfileGenerate"]
