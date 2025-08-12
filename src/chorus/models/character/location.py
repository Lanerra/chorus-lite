# src/chorus/models/character/location.py
"""Data model linking characters to locations."""

from __future__ import annotations

from uuid import UUID

from ..base_model import ChorusBaseModel as BaseModel


class CharacterLocation(BaseModel):
    """Join table linking a character with a location."""

    character_id: UUID
    location_id: UUID


__all__ = ["CharacterLocation"]
