# src/chorus/models/story/outline.py
"""Data model representing a story outline."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from ..chapter import Chapter
from ..mixins import IDMixin, TimestampsMixin


class Story(IDMixin, TimestampsMixin):
    """Hierarchical representation of a story."""

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",  # Allow extra fields from the LLM
        validate_assignment=True,
    )

    title: str = Field(..., min_length=1, description="Story title")
    concept_id: UUID | None = Field(
        default=None, description="Associated concept identifier"
    )
    genre: str | None = Field(default=None, description="Literary genre")
    themes: list[str] = Field(default_factory=list, description="Story themes")
    style_guide_id: UUID | None = Field(
        default=None, description="Optional style guide reference"
    )
    chapters: list[Chapter] = Field(default_factory=list, description="Story chapters")

    @model_validator(mode="before")
    @classmethod
    def _coerce_story_outline(cls, data: Any) -> Any:
        """Coerce LLM-generated outline formats into the expected structure."""
        if isinstance(data, dict):
            # If the LLM provides a 'story_outline' key, use it for chapters.
            if "story_outline" in data and "chapters" not in data:
                data["chapters"] = data["story_outline"]
            # If the LLM nests the outline inside a 'story' key.
            if "story" in data and isinstance(data["story"], dict):
                data.update(data["story"])
        return data

    @model_validator(mode="after")
    def _validate_chapters(self) -> Story:
        seen: set[int] = set()
        for chap in self.chapters:
            if chap.order_index is not None:
                if chap.order_index in seen:
                    raise ValueError("duplicate chapter order_index")
                seen.add(chap.order_index)
        return self


__all__ = ["Story"]
