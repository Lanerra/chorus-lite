# src/chorus/models/style_feedback.py
"""Feedback from the style editor."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from .base_model import ChorusBaseModel as BaseModel


class StyleFeedback(BaseModel):
    """Notes on style issues for a scene."""

    id: UUID | None = None
    version: int = 1
    scene_id: UUID
    notes: str = Field(..., min_length=1)
    created_at: datetime | None = None


__all__ = ["StyleFeedback"]
