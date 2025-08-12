# src/chorus/models/task/queue.py
"""Data model for queued tasks."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from ..mixins import IDMixin, TimestampsMixin


class TaskQueue(IDMixin, TimestampsMixin):
    """Representation of a queued task."""

    type: str = Field(..., min_length=1)
    payload: dict[str, Any] | None = None
    priority: int = 1
    dependencies: list[UUID] = Field(default_factory=list)
    assigned_to: list[UUID] = Field(default_factory=list)
    status: str = Field(..., min_length=1)


__all__ = ["TaskQueue"]
