# src/chorus/models/configuration.py
"""Configuration entry for agent prompts and parameters."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .base_model import ChorusBaseModel as BaseModel


class Configuration(BaseModel):
    """Key/value pair for agent configuration."""

    version: int = 1
    key: str = Field(..., min_length=1)
    value: Any
    created_at: datetime | None = None
    updated_at: datetime | None = None


__all__ = ["Configuration"]
