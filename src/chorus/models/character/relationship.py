# src/chorus/models/character/relationship.py
"""Data model for explicit character relationships."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from ..mixins import IDMixin, TimestampsMixin


class CharacterRelationship(IDMixin, TimestampsMixin):
    """Relationship between two characters."""

    from_character: UUID
    to_character: UUID
    relationship_type: str | None = None
    description: str | None = None
    strength: float | None = Field(default=None, ge=0, le=1)


__all__ = ["CharacterRelationship"]
