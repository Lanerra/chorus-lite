# src/chorus/models/entity.py
"""Models for named entity extraction results with minimal contracts."""

from __future__ import annotations

from enum import Enum

from pydantic import ConfigDict, Field

from .base_model import ChorusBaseModel as BaseModel


class EntityType(str, Enum):
    """Enumeration of possible entity categories."""

    CHARACTER = "CHARACTER"
    LOCATION = "LOCATION"
    ITEM = "ITEM"
    CONCEPT = "CONCEPT"


class Entity(BaseModel):
    """A named entity discovered in text.

    Minimal contract:
      - name: str
      - type: EntityType
      - description: Optional[str]
    Extra/unrecognized fields are forbidden.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type: EntityType
    description: str | None = None


__all__ = ["Entity", "EntityType"]
