# src/chorus/models/task/__init__.py
"""Task definitions for in-process task handling (LangGraph orchestrated)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from ..mixins import IDMixin, TimestampsMixin


class BaseTask(IDMixin, TimestampsMixin):
    """Base class for queued tasks."""

    task_type: str = Field(..., min_length=1)
    dependencies: list[UUID] = Field(default_factory=list)
    attempts: int = 0


class SceneBrief(BaseTask):
    """Brief for drafting a new scene (minimal contract)."""

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["scene_brief"] = "scene_brief"
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    characters: list[str] = Field(default_factory=list, min_length=1)
    intentions: list[str] = Field(default_factory=list, min_length=1)
    context: str | None = None  # Additional context


class RewriteTask(BaseTask):
    """Request to rewrite an existing scene.

    Includes the original ``draft`` text and editorial ``notes`` describing
    what needs improvement.
    """

    task_type: Literal["rewrite"] = "rewrite"
    scene_id: UUID
    notes: str = Field(..., min_length=1)
    draft: str = Field(..., min_length=1)


class ReviseTask(BaseTask):
    """Request to revise an existing scene based on feedback.

    Includes the original ``draft`` text and structured ``feedback``
    from reviewers.
    """

    task_type: Literal["revise"] = "revise"
    scene_id: UUID
    feedback: str = Field(..., min_length=1)
    draft: str = Field(..., min_length=1)


class IntegrateTask(BaseTask):
    """Request to integrate an approved scene into the Canon.

    Includes the scene ID and any additional metadata needed for integration.
    """

    task_type: Literal["integrate"] = "integrate"
    scene_id: UUID
    model: str = Field(..., min_length=1)


TASK_CLASSES: dict[str, type[BaseTask]] = {
    "scene_brief": SceneBrief,
    "rewrite": RewriteTask,
    "revise": ReviseTask,
    "integrate": IntegrateTask,
}

__all__ = [
    "BaseTask",
    "SceneBrief",
    "RewriteTask",
    "ReviseTask",
    "IntegrateTask",
    "TASK_CLASSES",
]
