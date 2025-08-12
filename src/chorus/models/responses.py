# src/chorus/models/responses.py
"""Pydantic models representing structured LLM responses."""

from __future__ import annotations

from pydantic import BaseModel, Field, RootModel

from .character import CharacterProfile
from .concept import Concept
from .entity import Entity
from .story import Scene, Story
from .task import SceneBrief
from .world_anvil import WorldAnvil


class ConceptList(RootModel[list[Concept]]):
    """List of :class:`Concept` items."""


class ConceptEvaluation(BaseModel):
    """Evaluation for a single concept."""

    index: int = Field(..., ge=0)
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1)


class ConceptEvaluationList(RootModel[list[ConceptEvaluation]]):
    """List of :class:`ConceptEvaluation` items."""


class SceneBriefList(RootModel[list[SceneBrief]]):
    """List of :class:`SceneBrief` items."""


class SceneList(RootModel[list[Scene]]):
    """List of :class:`Scene` items."""


class StoryList(RootModel[list[Story]]):
    """List of :class:`Story` items."""


class CharacterProfileList(RootModel[list[CharacterProfile]]):
    """List of :class:`CharacterProfile` items."""


class WorldAnvilList(RootModel[list[WorldAnvil]]):
    """List of :class:`WorldAnvil` items."""


class InterventionList(RootModel[list[str]]):
    """List of intervention strings."""


class StringResponse(RootModel[str]):
    """Simple string response."""


class IntResponse(RootModel[int]):
    """Integer response value."""


class EntityList(RootModel[list[Entity]]):
    """RootModel[List[Entity]] with no extra wrapper."""


class ContinuityFeedback(BaseModel):
    """Feedback on scene continuity issues."""

    issues: list[str] = Field(
        default_factory=list, description="List of continuity issues found"
    )
    score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Continuity score from 0.0 to 1.0"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )


class StyleFeedback(BaseModel):
    """Feedback on scene style and prose quality."""

    issues: list[str] = Field(
        default_factory=list, description="List of style issues found"
    )
    score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Style score from 0.0 to 1.0"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )


class StoryFeedback(BaseModel):
    """General story feedback model."""

    issues: list[str] = Field(
        default_factory=list, description="List of story issues found"
    )
    score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Story score from 0.0 to 1.0"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )


class ValidationResult(BaseModel):
    """Result of story or content validation."""

    is_valid: bool = Field(..., description="Whether the validation passed")
    errors: list[str] | None = Field(
        default=None, description="List of validation errors"
    )
    warnings: list[str] | None = Field(
        default=None, description="List of validation warnings"
    )
    suggestions: list[str] | None = Field(
        default=None, description="List of improvement suggestions"
    )


__all__ = [
    "ConceptList",
    "ConceptEvaluation",
    "ConceptEvaluationList",
    "SceneBriefList",
    "CharacterProfileList",
    "WorldAnvilList",
    "InterventionList",
    "StringResponse",
    "IntResponse",
    "SceneList",
    "StoryList",
    "ContinuityFeedback",
    "StyleFeedback",
    "StoryFeedback",
    "ValidationResult",
]
