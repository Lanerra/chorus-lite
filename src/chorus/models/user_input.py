# src/chorus/models/user_input.py
"""User input models for Chorus-Lite's streamlined architecture."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from .base_model import ChorusBaseModel as BaseModel


class NovelConceptModel(BaseModel):
    """High-level concept for the novel."""
    
    title: str = Field(..., min_length=1)
    genre: Optional[str] = None
    setting: Optional[str] = None
    theme: Optional[str] = None


class RelationshipModel(BaseModel):
    """Relationship details for a character."""
    
    name: Optional[str] = None
    status: Optional[str] = None
    details: Optional[str] = None
    type: Optional[str] = None  # friendship, rivalry, mentorship, etc.


class ProtagonistModel(BaseModel):
    """Primary character information."""
    
    name: str
    description: Optional[str] = None
    # Simplified traits (no complex trait system in Chorus-Lite)
    traits: List[str] = []
    role: Optional[str] = None


class CharacterGroupModel(BaseModel):
    """Container for characters provided in user input."""
    
    protagonist: Optional[ProtagonistModel] = None
    antagonist: Optional[ProtagonistModel] = None
    supporting_characters: List[ProtagonistModel] = []
    character_groups: Dict[str, List[str]] = {}  # e.g., {"family": ["John", "Mary"]}


class KeyLocationModel(BaseModel):
    """A single location within the setting."""
    
    name: str
    description: Optional[str] = None
    atmosphere: Optional[str] = None


class SettingModel(BaseModel):
    """Setting information for the story world."""
    
    primary_setting_overview: Optional[str] = None
    key_locations: List[KeyLocationModel] = []
    world_rules: List[str] = []


class PlotElementsModel(BaseModel):
    """Major plot elements provided by the user."""
    
    plot_points: List[str] = []
    major_twists: List[str] = []


class UserStoryInputModel(BaseModel):
    """Top-level structure for user story input."""
    
    novel_concept: Optional[NovelConceptModel] = None
    protagonist: Optional[ProtagonistModel] = None
    antagonist: Optional[ProtagonistModel] = None
    characters: Optional[CharacterGroupModel] = None
    plot_elements: Optional[PlotElementsModel] = None
    setting: Optional[SettingModel] = None
    world_details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")
