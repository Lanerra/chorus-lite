# src/chorus/models/mixins.py
"""Common reusable mixin models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .base_model import ChorusBaseModel


class IDMixin(ChorusBaseModel):
    """Mixin that provides a unique identifier."""

    id: UUID | None = Field(default=None)
    version: int = 1

    @model_validator(mode="after")
    def _ensure_id(self) -> IDMixin:
        if self.id is None:
            self.id = uuid4()
        return self


class TimestampsMixin(ChorusBaseModel):
    """Mixin that adds creation and update timestamps."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default_factory=lambda: datetime.now(UTC))


__all__ = ["IDMixin", "TimestampsMixin"]
