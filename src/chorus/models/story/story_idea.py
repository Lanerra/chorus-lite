# src/chorus/models/story/story_idea.py
from __future__ import annotations

from src.chorus.models.base_model import BaseModel
from typing import Optional
from src.chorus.models.story.outline import Story
"""Model for story ideas."""

from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import Field

from src.chorus.models.base_model import BaseModel

class StoryIdea(BaseModel):
    """Model representing a story idea."""

    __tablename__ = "story_ideas"

    id: int = Field(default=None, primary_key=True, index=True)
    title: str = Field(default=None, description="Title of the story idea")
    description: str = Field(default=None, description="Description of the story idea")
    genre: str | None = Field(default=None, description="Genre of the story idea")
    mood: str | None = Field(default=None, description="Mood of the story idea")
    target_audience: str | None = Field(default=None, description="Target audience for the story idea")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    story_id: int | None = Field(default=None, foreign_key="stories.id", description="ID of the associated story")
    
    def story(self) -> "Story" | None:
        """Lazy load the associated story."""
        return Story.get(self.story_id) if self.story_id else None

    def __repr__(self):
        return f"<StoryIdea(id={self.id}, title='{self.title}')>"
