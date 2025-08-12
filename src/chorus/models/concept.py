# src/chorus/models/concept.py
"""Data model representing a high-level story concept."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from .base_model import ChorusBaseModel as BaseModel


class Concept(BaseModel):
    """A generated idea for the overall narrative vision (minimal contract).

    Tightening:
      - extra="forbid"
      - At least one of title or logline is required.
      - Other optional fields are preserved.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID | None = None
    version: int = 1
    title: str | None = None
    logline: str | None = Field(default=None)
    theme: str | None = None
    conflict: str | None = None
    protagonist: str | None = None
    hook: str | None = None
    genre: str | None = None
    tone: str | None = None
    setting: str | None = None
    mood: str | None = None
    created_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _use_summary_for_logline(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "summary" in data and "logline" not in data:
                data["logline"] = data["summary"]
        return data

    @model_validator(mode="after")
    def _ensure_title_or_logline(self) -> Concept:
        title_ok = bool(self.title and self.title.strip())
        logline_ok = bool(self.logline and self.logline.strip())
        if not (title_ok or logline_ok):
            raise ValueError("at least one of 'title' or 'logline' must be present")
        return self


__all__ = ["Concept"]
