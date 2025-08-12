# src/chorus/models/world_item.py
"""Data model for world elements in Chorus-Lite's lightweight approach."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from .base_model import ChorusBaseModel as BaseModel


class WorldItem(BaseModel):
    """Structured world element information for Chorus-Lite's lightweight approach."""
    
    # Core identity
    id: str
    category: str
    name: str
    
    # Temporal tracking
    created_chapter: int = 0
    elaboration_history: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Properties and metadata
    properties: Dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    lore: List[str] = Field(default_factory=list)
    
    # Meta information
    version: int = 1

    @field_validator("elaboration_history", mode="before")
    @classmethod
    def _validate_elaboration_history(cls, v: Dict[str, List[str]] | None) -> Dict[str, List[str]] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("elaboration_history must be a dictionary")
        # Validate each chapter key and its list of events
        for chapter, events in v.items():
            if not isinstance(chapter, str) or not chapter:
                raise ValueError("Chapter keys must be non-empty strings")
            if not isinstance(events, list):
                raise ValueError(f"Events for chapter {chapter} must be a list")
            # Validate each event
            for event in events:
                if not isinstance(event, str) or not event.strip():
                    raise ValueError(f"Event '{event}' is invalid")
        return v

    @field_validator("properties", mode="before")
    @classmethod
    def _validate_properties(cls, v: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("properties must be a dictionary")
        # Ensure all values are serializable
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Property key '{key}' must be a string")
            # Accept any type but ensure it's serializable
            try:
                str(value)
            except Exception:
                raise ValueError(f"Property value for '{key}' is not serializable")
        return v


class WorldElementUpdate(BaseModel):
    """Minimal contract for updating world elements in Chorus-Lite."""
    
    id: str
    name: str | None = None
    description: str | None = None
    category: str | None = None
    
    # Updates
    properties: Dict[str, Any] | None = None
    lore: List[str] | None = None

    model_config = ConfigDict(extra="forbid")
