# src/chorus/models/story/evaluation.py
"""Models for evaluating a story outline."""

from __future__ import annotations

from pydantic import Field

from ..base_model import ChorusBaseModel as BaseModel
from .outline import Story


class OutlineEvaluation(BaseModel):
    """Evaluation result for a story outline."""

    approved: bool = Field(
        ..., description="True if the outline meets quality standards"
    )
    notes: list[str] = Field(default_factory=list, description="Feedback notes")
    revised_story: Story | None = Field(
        default=None,
        description="Optional revised outline if changes are needed",
    )


__all__ = ["OutlineEvaluation"]
