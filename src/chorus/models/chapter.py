# src/chorus/models/chapter.py
"""Data model for story chapters."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from .base_model import ChorusBaseModel as BaseModel
from .task import SceneBrief


class Chapter(BaseModel):
    """Representation of a narrative chapter."""

    # Tighten to minimal contract for submodel
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    description: str | None = None
    order_index: int | None = None
    structure_notes: dict[str, Any] | None = None
    created_at: datetime | None = None
    scenes: list[SceneBrief] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_scenes(self) -> Chapter:
        seen: set[int] = set()
        for scene in self.scenes:
            chap_id = getattr(scene, "chapter_id", None)
            if chap_id is not None and self.id is not None:
                if chap_id != self.id:
                    raise ValueError("scene.chapter_id mismatch")
            num = getattr(scene, "scene_number", None)
            if num is not None:
                if num in seen:
                    raise ValueError("duplicate scene_number in chapter")
                seen.add(num)
        return self


__all__ = ["Chapter"]
