# src/chorus/models/sqlalchemy_models.py
"""SQLAlchemy ORM models for the Canon stored in PostgreSQL."""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class ApprovalLogSQL(Base):
    """Audit log entry for scene approval events.

    Each record tracks a reviewer's decision on a scene and stores any
    associated feedback.  This table allows the system to maintain a
    historical record of approvals, rejections and the rationale
    behind them.
    """

    __tablename__ = "approval_log"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    scene_id = Column(UUID(as_uuid=True), ForeignKey("scene.id"), nullable=False)
    reviewer = Column(Text)
    status = Column(String(50), nullable=False)
    feedback: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    timestamp = Column(TIMESTAMP, server_default=func.now())


class ChapterSQL(Base):
    """A chapter in a story.

    Chapters group together a sequence of scenes and provide metadata
    such as the title, an optional description, a user-defined order
    index and any structural notes.  A chapter may have zero or more
    :class:`SceneSQL` instances associated with it via the
    ``scenes`` relationship.
    """

    __tablename__ = "chapter"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    title = Column(Text, nullable=False)
    description = Column(Text)
    order_index = Column(Integer)
    structure_notes = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
    scenes: Mapped[list[SceneSQL]] = relationship(
        "SceneSQL",
        back_populates="chapter",
    )  # type: ignore[assignment]


class CharacterLocationSQL(Base):
    """Association between a character and a location.

    This join table links character profiles to locations in the
    world, enabling many-to-many relationships.  It is used when
    characters are known to inhabit or be associated with particular
    :class:`WorldAnvilSQL` entries.
    """

    __tablename__ = "character_location"
    character_id = Column(
        UUID(as_uuid=True), ForeignKey("character_profile.id"), primary_key=True
    )
    location_id = Column(
        UUID(as_uuid=True), ForeignKey("world_anvil.id"), primary_key=True
    )


class CharacterProfileSQL(Base):
    """A character's profile and biography.

    This table stores the key attributes and personal history of
    characters in the story world.  Fields cover both factual data
    (such as name, age, species, rank and dates) and narrative
    constructs (beliefs, desires, motivations, fatal flaw, arc and
    voice).  Timestamp columns track when the profile was created and
    last updated.
    """

    __tablename__ = "character_profile"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name = Column(Text, nullable=False, unique=True)
    full_name = Column(Text)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    gender = Column(Text)
    age = Column(Integer)
    birth_date = Column(Date)
    death_date = Column(Date)
    species = Column(Text)
    role = Column(Text)
    rank = Column(Text)
    backstory = Column(Text)
    beliefs: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    desires: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    intentions: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    motivations: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    fatal_flaw = Column(Text)
    arc = Column(Text)
    voice = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, onupdate=func.now())


class CharacterRelationshipSQL(Base):
    """Directed relationship between two characters.

    Records how two characters are connected, including the type of
    relationship (e.g., friend, enemy, mentor), an optional free-form
    description and a numeric ``strength`` value reflecting the
    intensity of the relationship.  Both endpoints reference
    :class:`CharacterProfileSQL`.
    """

    __tablename__ = "character_relationship"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    from_character = Column(
        UUID(as_uuid=True), ForeignKey("character_profile.id"), nullable=False
    )
    to_character = Column(
        UUID(as_uuid=True), ForeignKey("character_profile.id"), nullable=False
    )
    relationship_type = Column(Text)
    description = Column(Text)
    strength = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())


class ConceptSQL(Base):
    """High-level concept definition for a story.

    Concepts capture the abstract elements of a narrative such as the
    logline, theme, central conflict, protagonist, hook, genre, tone,
    setting and mood.  A :class:`StorySQL` may link to one of these
    records to anchor its creative direction.
    """

    __tablename__ = "concept"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    logline = Column(Text, nullable=False)
    theme = Column(Text)
    conflict = Column(Text)
    protagonist = Column(Text)
    hook = Column(Text)
    genre = Column(Text)
    tone = Column(Text)
    setting = Column(Text)
    mood = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class ContinuityFeedbackSQL(Base):
    """Feedback about continuity issues in a scene.

    During review, readers may identify inconsistencies, plot holes or
    other continuity problems within a scene.  These notes are stored
    here and linked back to the offending scene via ``scene_id`` for
    later analysis or revision.
    """

    __tablename__ = "continuity_feedback"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    scene_id = Column(UUID(as_uuid=True), ForeignKey("scene.id"), nullable=False)
    notes = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class EventSQL(Base):
    """A discrete event in the narrative universe.

    Events represent notable happenings—battles, ceremonies, turning
    points—that characters may refer to.  They consist of a unique
    name, an optional description and a creation timestamp.
    """

    __tablename__ = "event"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class ItemSQL(Base):
    """An important item or object.

    Items can be weapons, artifacts, tools or any significant objects
    that appear in the story.  Each record stores a unique name, an
    optional description and a creation timestamp.
    """

    __tablename__ = "item"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class LoreSQL(Base):
    """SQLAlchemy model for miscellaneous lore entries."""

    __tablename__ = "lore"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    key = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class LocationSQL(Base):
    """A named location within the story world.

    Locations are places—cities, planets, rooms—where scenes can take
    place.  Each has a unique name, an optional description and a
    creation timestamp.
    """

    __tablename__ = "location"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class OrganizationSQL(Base):
    """An organization or faction in the story world.

    Organizations may include guilds, governments, companies or other
    groups.  Each record contains a unique name, an optional
    description and a creation timestamp.
    """

    __tablename__ = "organization"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class SceneSQL(Base):
    """A scene within a chapter.

    Scenes are the fundamental storytelling units.  This table stores
    metadata and content for a scene: title, description, full text,
    vector embedding (for similarity search), status (e.g., draft or
    approved), scene number, setting, the characters present and
    associated world element via ``location_id``.  Scenes belong to
    chapters via ``chapter_id`` and record when they were created and
    last updated.
    """

    __tablename__ = "scene"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    title = Column(Text, nullable=False)
    description = Column(Text)
    text = Column(Text)
    embedding = Column(Vector(768), nullable=True)  # type: ignore[var-annotated]
    status = Column(String(50), nullable=False, index=True)
    scene_number = Column(Integer)
    setting = Column(Text)
    characters: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    location_id = Column(UUID(as_uuid=True), ForeignKey("world_anvil.id"))
    chapter_id = Column(UUID(as_uuid=True), ForeignKey("chapter.id"))
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, onupdate=func.now())
    chapter: Mapped[ChapterSQL | None] = relationship(
        "ChapterSQL",
        back_populates="scenes",
    )  # type: ignore[assignment]


class StoryFeedbackSQL(Base):
    """General feedback note for a scene.

    Distinct from style or continuity feedback, story feedback
    captures readers' observations and suggestions about the content of
    a scene.  Each note references the associated scene and includes
    the feedback text and a timestamp.
    """

    __tablename__ = "story_feedback"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    scene_id = Column(UUID(as_uuid=True), ForeignKey("scene.id"))
    notes = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class StorySQL(Base):
    """A top-level story record.

    Each story has a title and may link to a :class:`ConceptSQL` record
    that provides its creative direction.  Stories are a container for
    chapters, though the chapters are stored in a separate table.
    """

    __tablename__ = "story"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    title = Column(Text, nullable=False)
    concept_id = Column(UUID(as_uuid=True), ForeignKey("concept.id"))
    created_at = Column(TIMESTAMP, server_default=func.now())


class TaskQueueSQL(Base):
    """A queued task in the internal worker system.

    Tasks represent units of work waiting to be processed by workers
    (e.g., scene generation, revision).  They include a type, an
    arbitrary JSON ``payload``, a ``priority``, a list of dependency
    task identifiers, an optional ``assigned_to`` worker, the current
    status, and timestamps.
    """

    __tablename__ = "task_queue"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    type = Column(String(50), nullable=False)
    payload = Column(JSON)
    priority = Column(Integer, default=1)
    dependencies: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=[]
    )
    assigned_to = Column(UUID(as_uuid=True))
    status = Column(String(50), default="QUEUED")
    created_at = Column(TIMESTAMP, server_default=func.now())


class WorldAnvilSQL(Base):
    """A world-building element.

    World Anvil entries represent locations, organizations or items
    imported from or modeled after WorldAnvil.  Fields include
    ``name``, ``description``, ``category``, a list of ``tags``, an
    optional ``location_type``, optional ``ruling_power`` and a JSON
    blob of ``cultural_notes``.  Creation and update timestamps are
    tracked.
    """

    __tablename__ = "world_anvil"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    category = Column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=[])
    location_type = Column(Text)
    ruling_power = Column(Text)
    cultural_notes = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, onupdate=func.now())


class ConfigurationSQL(Base):
    """Arbitrary configuration key/value.

    This simple key/value table is used to store runtime configuration
    for the system.  Values are stored as JSON to allow complex
    structures and are keyed by a unique string.  Timestamps record
    when the configuration was created or last updated.
    """

    __tablename__ = "configuration"
    key = Column(String(100), primary_key=True)
    value = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, onupdate=func.now())
