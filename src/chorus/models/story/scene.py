# src/chorus/models/story/scene.py
"""Data model representing a scene in the writing pipeline."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import ConfigDict, Field, ValidationInfo, model_validator

from ..mixins import IDMixin, TimestampsMixin


class SceneStatus(Enum):
    """Possible workflow states for a scene.

    Simplified from 7 to 5 statuses as part of single-pass workflow conversion.
    Legacy statuses are mapped as follows:
    - in_review_continuity, in_review_style → drafting
    - rejected_continuity, rejected_style → rejected
    """

    QUEUED = "queued"
    DRAFTING = "drafting"
    IN_REVIEW = "in_review"
    REJECTED = "rejected"
    APPROVED = "approved"


class Scene(IDMixin, TimestampsMixin):
    """Description of a scene being written."""

    # Story root remains ignore; submodels are tightened elsewhere.
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, description="Human-readable title")
    description: str | None = Field(
        default=None, description="Optional summary of the scene"
    )
    text: str = Field("", description="Generated scene text")
    embedding: list[float] = Field(
        default_factory=list, description="Vector embedding of the text"
    )
    status: SceneStatus = Field(
        default=SceneStatus.QUEUED,
        description="Workflow status",
    )
    scene_number: int | None = Field(
        default=None, description="Ordering number within the chapter"
    )
    setting: str | None = Field(default=None, description="Scene setting")
    characters: list[UUID] = Field(
        default_factory=list, description="Characters present"
    )
    location_id: UUID | None = Field(
        default=None, description="Canonical location identifier"
    )
    chapter_id: UUID | None = Field(
        default=None, description="Owning chapter identifier"
    )
    # Timestamps provided by mixin

    @model_validator(mode="after")
    def _validate_characters(self, info: ValidationInfo) -> Scene:
        allowed: set[UUID] | None = None
        if info.context is not None:
            allowed = info.context.get("character_ids")  # type: ignore[assignment]
        if allowed is not None:
            invalid = [cid for cid in self.characters if cid not in allowed]
            if invalid:
                raise ValueError(f"invalid character ids: {invalid}")
        return self


__all__ = ["Scene", "SceneStatus"]
