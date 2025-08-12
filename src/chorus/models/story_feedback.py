# src/chorus/models/story_feedback.py
"""High-level feedback on the story."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import Field

from .base_model import ChorusBaseModel as BaseModel


class StoryFeedbackType(str, Enum):
    """Types of story feedback."""
    
    CONTINUITY = "continuity"
    STYLE = "style"
    PLOT = "plot"
    CHARACTER = "character"
    WORLD_BUILDING = "world_building"


class StoryFeedback(BaseModel):
    """General comments about story quality."""

    id: UUID | None = None
    version: int = 1
    scene_id: UUID | None = None
    feedback_type: StoryFeedbackType = StoryFeedbackType.CONTINUITY
    notes: str = Field(..., min_length=1)
    created_at: datetime | None = None


__all__ = ["StoryFeedback", "StoryFeedbackType"]
