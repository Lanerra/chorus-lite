# src/chorus/models/approval_log.py
"""Data model for scene approval history."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from .base_model import ChorusBaseModel as BaseModel


class ApprovalLog(BaseModel):
    """Record of scene approval and rejection."""

    id: UUID | None = None
    version: int = 1
    scene_id: UUID
    reviewer: str | None = None
    status: str = Field(..., min_length=1)
    feedback: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None


__all__ = ["ApprovalLog"]
