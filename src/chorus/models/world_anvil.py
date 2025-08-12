# src/chorus/models/world_anvil.py
"""Data model for world building elements."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from .base_model import ChorusBaseModel as BaseModel


class WorldAnvil(BaseModel):
    """Representation of a world element."""

    id: UUID | None = None
    version: int = 1
    name: str = Field(..., min_length=1)
    description: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    location_type: str | None = None
    ruling_power: str | None = None
    cultural_notes: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


__all__ = ["WorldAnvil"]
