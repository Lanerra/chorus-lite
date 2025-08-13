# src/chorus/canon/crud.py
"""High-level create/read/update/delete operations used by agents."""

# mypy: disable-error-code=arg-type

from __future__ import annotations

from typing import Any
from uuid import UUID
import time

import yaml
from psycopg.types.json import Jsonb
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from chorus.canon.db import _maybe_await
from chorus.core.logs import get_event_logger, EventType, Priority
from chorus.models import (
    ApprovalLog,
    Chapter,
    CharacterProfile,
    CharacterRelationship,
    Concept,
    ContinuityFeedback,
    Scene,
    SceneStatus,
    Story,
    StoryFeedback,
    TaskQueue,
    WorldAnvil,
)

from .queries import store_scene_text_conn

# Initialize EventLogger for database operations
event_logger = get_event_logger()


async def set_configuration_conn(session: AsyncSession, key: str, value: Any) -> None:
    """Insert or update configuration ``key`` to ``value`` using ``session``."""
    
    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        f"Setting configuration key: {key}",
        Priority.NORMAL,
        metadata={"operation": "upsert", "table": "configuration", "key": key, "value_type": type(value).__name__}
    )

    try:
        await session.execute(
            sa_text(
                "INSERT INTO configuration (key, value) VALUES (:key, :val) "
                "ON CONFLICT (key) DO UPDATE SET value = :val, updated_at = NOW()"
            ),
            {"key": key, "val": Jsonb(value)},
        )
        
        duration = time.time() - start_time
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Successfully set configuration {key}",
            Priority.NORMAL,
            metadata={"operation": "upsert", "table": "configuration", "duration": duration, "success": True}
        )
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"Setting configuration {key}",
            metadata={"operation": "upsert", "table": "configuration", "duration": duration, "key": key}
        )
        raise


async def set_configuration(key: str, value: Any) -> None:
    """Insert or update configuration ``key`` with ``value``."""

    await event_logger.log(
        EventType.DATABASE_OPERATION,
        f"Setting configuration with new session: {key}",
        Priority.NORMAL,
        metadata={"operation": "set_configuration", "key": key, "session_management": True}
    )

    from chorus.canon.db import get_pg

    async with get_pg() as session:
        await set_configuration_conn(session, key, value)
        await session.commit()
        
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Configuration {key} committed successfully",
            Priority.NORMAL,
            metadata={"operation": "set_configuration", "key": key, "committed": True}
        )


async def get_configuration(session: AsyncSession, key: str) -> Any | None:
    """Return configuration value for ``key``."""

    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        f"Retrieving configuration for key: {key}",
        Priority.NORMAL,
        metadata={"operation": "select", "table": "configuration", "key": key}
    )

    try:
        result = await session.execute(
            sa_text("SELECT value FROM configuration WHERE key = :key"), {"key": key}
        )
        row = await _maybe_await(result.fetchone())
        
        duration = time.time() - start_time
        found = row is not None
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Configuration lookup for {key}: {'found' if found else 'not found'}",
            Priority.NORMAL,
            metadata={"operation": "select", "table": "configuration", "duration": duration, "found": found}
        )
        
        return row[0] if row else None
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"Getting configuration {key}",
            metadata={"operation": "select", "table": "configuration", "duration": duration, "key": key}
        )
        raise


async def load_configuration_file(path: str) -> None:
    """Load YAML file at ``path`` into the configuration table."""

    from chorus.canon.db import get_pg

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    async with get_pg() as session:
        for k, v in data.items():
            await set_configuration_conn(session, k, v)
        await session.commit()


async def record_story_feedback(scene_id: UUID | int | None, notes: str) -> None:
    """Store high-level story feedback."""

    from chorus.canon.db import get_pg

    async with get_pg() as session:
        await session.execute(
            sa_text(
                "INSERT INTO story_feedback (scene_id, notes) VALUES (:sid, :notes)"
            ),
            {"sid": scene_id if scene_id is not None else None, "notes": notes},
        )
        await session.commit()


async def create_story_feedback(
    session: AsyncSession, feedback_data: StoryFeedback
) -> StoryFeedback:
    """Insert ``feedback_data`` and return the populated model."""

    result = await session.execute(
        sa_text(
            "INSERT INTO story_feedback (scene_id, notes) VALUES (:sid, :notes) RETURNING id"
        ),
        {"sid": feedback_data.scene_id, "notes": feedback_data.notes},
    )
    row = await _maybe_await(result.fetchone())
    if row:
        feedback_data.id = row[0]
    await session.commit()
    return feedback_data


async def get_feedback_for_scene(session: AsyncSession, scene_id: str | UUID) -> dict:
    """Return all feedback linked to ``scene_id``."""

    cont_result = await session.execute(
        sa_text(
            "SELECT id, scene_id, notes, created_at FROM continuity_feedback WHERE scene_id = :sid ORDER BY created_at"
        ),
        {"sid": scene_id},
    )
    cont_rows = cont_result.all()
    continuity = [
        ContinuityFeedback(id=r[0], scene_id=r[1], notes=r[2], created_at=r[3])
        for r in cont_rows
    ]

    story_result = await session.execute(
        sa_text(
            "SELECT id, scene_id, notes, created_at FROM story_feedback WHERE scene_id = :sid ORDER BY created_at"
        ),
        {"sid": scene_id},
    )
    story_rows = story_result.all()
    story = [
        StoryFeedback(id=r[0], scene_id=r[1], notes=r[2], created_at=r[3])
        for r in story_rows
    ]

    return {"continuity": continuity, "story": story}


async def create_approval_log(
    session: AsyncSession, log_data: ApprovalLog
) -> ApprovalLog:
    """Insert ``log_data`` into ``approval_log`` and return the populated model."""

    result = await session.execute(
        sa_text(
            "INSERT INTO approval_log (scene_id, reviewer, status, feedback) VALUES (:sid, :reviewer, :status, :feedback) RETURNING id, timestamp"
        ),
        {
            "sid": log_data.scene_id,
            "reviewer": log_data.reviewer,
            "status": log_data.status,
            "feedback": log_data.feedback,
        },
    )
    row = await _maybe_await(result.fetchone())
    if row:
        log_data.id = row[0]
        log_data.timestamp = row[1]
    await session.commit()
    return log_data


async def create_task_log(session: AsyncSession, task_data: TaskQueue) -> TaskQueue:
    """Insert ``task_data`` into ``task_queue`` and return the populated model."""

    result = await session.execute(  # type: ignore[arg-type]
        sa_text(
            "INSERT INTO task_queue (type, payload, priority, dependencies, assigned_to, status) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, created_at"
        ),
        (
            task_data.type,
            Jsonb(task_data.payload) if task_data.payload is not None else None,
            task_data.priority,
            task_data.dependencies,
            task_data.assigned_to,
            task_data.status,
        ),
    )
    row = await _maybe_await(result.fetchone())
    if row:
        task_data.id = row[0]
        task_data.created_at = row[1]
    await session.commit()
    return task_data


async def create_character_profile(
    session: AsyncSession, profile: CharacterProfile
) -> CharacterProfile:
    """Insert ``profile`` and return it with the new ID."""

    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        f"Creating character profile: {profile.name}",
        Priority.NORMAL,
        metadata={
            "operation": "upsert",
            "table": "character_profile",
            "character_name": profile.name,
            "role": profile.role,
            "has_backstory": bool(profile.backstory)
        }
    )

    try:
        result = await session.execute(
            sa_text(
                "INSERT INTO character_profile (name, full_name, aliases, gender, age, birth_date, death_date, species, role, rank, backstory, beliefs, desires, intentions, motivations, fatal_flaw, arc, voice) "
                "VALUES (:name, :full_name, :aliases, :gender, :age, :birth_date, :death_date, :species, :role, :rank, :backstory, :beliefs, :desires, :intentions, :motivations, :fatal_flaw, :arc, :voice) "
                "ON CONFLICT (name) DO UPDATE SET full_name = EXCLUDED.full_name, backstory = EXCLUDED.backstory, beliefs = EXCLUDED.beliefs, desires = EXCLUDED.desires, intentions = EXCLUDED.intentions, motivations = EXCLUDED.motivations, fatal_flaw = EXCLUDED.fatal_flaw, arc = EXCLUDED.arc, voice = EXCLUDED.voice, updated_at = NOW() RETURNING id, created_at"
            ),
            {
                "name": profile.name,
                "full_name": profile.full_name,
                "aliases": profile.aliases,
                "gender": profile.gender,
                "age": profile.age,
                "species": profile.species,
                "role": profile.role,
                "backstory": profile.backstory,
                "fatal_flaw": profile.fatal_flaw,

            },
        )
        row = await _maybe_await(result.fetchone())
        if row:
            profile.id, profile.created_at = row
        
        duration = time.time() - start_time
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Successfully created character profile: {profile.name} (ID: {profile.id})",
            Priority.NORMAL,
            metadata={
                "operation": "upsert",
                "table": "character_profile",
                "duration": duration,
                "character_id": str(profile.id),
                "character_name": profile.name,
                "success": True
            }
        )
        
        return profile
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"Creating character profile {profile.name}",
            metadata={
                "operation": "upsert",
                "table": "character_profile",
                "duration": duration,
                "character_name": profile.name
            }
        )
        raise


async def get_character_profile(
    session: AsyncSession, profile_id: str | UUID
) -> CharacterProfile | None:
    """Return a :class:`CharacterProfile` by ID."""

    result = await session.execute(
        sa_text(
            "SELECT id, name, full_name, aliases, gender, age, birth_date, death_date, species, role, rank, backstory, beliefs, desires, intentions, motivations, fatal_flaw, created_at, updated_at FROM character_profile WHERE id = :pid"
        ),
        {"pid": profile_id},
    )
    row = await _maybe_await(result.fetchone())
    if row is None:
        return None
    (
        cid,
        name,
        full_name,
        aliases,
        gender,
        age,
        species,
        role,
        backstory,
        fatal_flaw,
        created_at,
        updated_at,
    ) = row
    return CharacterProfile(
        id=cid,
        name=name,
        full_name=full_name,
        aliases=list(aliases) if aliases is not None else [],
        gender=gender,
        age=age,
        species=species,
        role=role,
        backstory=backstory,
        fatal_flaw=fatal_flaw,
        created_at=created_at,
        updated_at=updated_at,
    )


async def update_character_profile(
    session: AsyncSession, profile: CharacterProfile
) -> None:
    """Persist updates to ``profile``."""

    await session.execute(
        sa_text(
            "UPDATE character_profile SET name = :name, full_name = :full_name, aliases = :aliases, gender = :gender, age = :age, birth_date = :birth_date, death_date = :death_date, species = :species, role = :role, rank = :rank, backstory = :backstory, beliefs = :beliefs, desires = :desires, intentions = :intentions, motivations = :motivations, fatal_flaw = :fatal_flaw, arc = :arc, voice = :voice, updated_at = NOW() WHERE id = :id"
        ),
        {
            "name": profile.name,
            "full_name": profile.full_name,
            "aliases": profile.aliases,
            "gender": profile.gender,
            "age": profile.age,
            "birth_date": profile.birth_date,
            "death_date": profile.death_date,
            "species": profile.species,
            "role": profile.role,
            "backstory": profile.backstory,
            "fatal_flaw": profile.fatal_flaw,
            "id": profile.id,
        },
    )


async def delete_character_profile(
    session: AsyncSession, profile_id: str | UUID
) -> None:
    """Remove a character profile from the database."""

    await session.execute(
        sa_text("DELETE FROM character_profile WHERE id = :pid"), {"pid": profile_id}
    )


async def create_character_relationship(
    session: AsyncSession, rel: CharacterRelationship
) -> CharacterRelationship:
    """Insert ``rel`` and return it with the new ID."""

    result = await session.execute(
        sa_text(
            "INSERT INTO character_relationship (from_character, to_character, "
            "relationship_type, description, strength) VALUES ("
            ":from_character, :to_character, :relationship_type, :description, "
            ":strength) RETURNING id, created_at"
        ),
        {
            "from_character": rel.from_character,
            "to_character": rel.to_character,
            "relationship_type": rel.relationship_type,
            "description": rel.description,
            "strength": rel.strength,
        },
    )
    row = await _maybe_await(result.fetchone())
    if row:
        rel.id, rel.created_at = row
    return rel


async def create_world_anvil(session: AsyncSession, wa: WorldAnvil) -> WorldAnvil:
    """Insert ``wa`` and return it with the new ID."""

    result = await session.execute(
        sa_text(
            "INSERT INTO world_anvil (name, description, category, tags, "
            "location_type, ruling_power, cultural_notes) "
            "VALUES (:name, :description, :category, :tags, :location_type, "
            ":ruling_power, :cultural_notes) "
            "ON CONFLICT (name) DO UPDATE SET "
            "description = EXCLUDED.description, category = EXCLUDED.category, "
            "tags = EXCLUDED.tags, location_type = EXCLUDED.location_type, "
            "ruling_power = EXCLUDED.ruling_power, cultural_notes = "
            "EXCLUDED.cultural_notes, updated_at = NOW() "
            "RETURNING id, created_at, updated_at"
        ),
        {
            "name": wa.name,
            "description": wa.description,
            "category": wa.category,
            "tags": wa.tags,
            "location_type": wa.location_type,
            "ruling_power": wa.ruling_power,
            "cultural_notes": Jsonb(wa.cultural_notes)
            if wa.cultural_notes is not None
            else None,
        },
    )
    row = await _maybe_await(result.fetchone())
    if row:
        wa.id, wa.created_at, wa.updated_at = row
    return wa


async def get_world_anvil(
    session: AsyncSession, wa_id: str | UUID
) -> WorldAnvil | None:
    """Return :class:`WorldAnvil` by ID."""

    result = await session.execute(
        sa_text(
            "SELECT id, name, description, category, tags, location_type, "
            "ruling_power, cultural_notes, created_at, updated_at FROM world_anvil "
            "WHERE id = :wa_id"
        ),
        {"wa_id": wa_id},
    )
    row = await _maybe_await(result.fetchone())
    if row is None:
        return None
    (
        wid,
        name,
        description,
        category,
        tags,
        location_type,
        ruling_power,
        cultural_notes,
        created_at,
        updated_at,
    ) = row
    return WorldAnvil(
        id=wid,
        name=name,
        description=description,
        category=category,
        tags=list(tags) if tags is not None else [],
        location_type=location_type,
        ruling_power=ruling_power,
        cultural_notes=cultural_notes,
        created_at=created_at,
        updated_at=updated_at,
    )


async def update_world_anvil(session: AsyncSession, wa: WorldAnvil) -> None:
    """Persist updates to ``wa``."""

    await session.execute(
        sa_text(
            "UPDATE world_anvil SET name = :name, description = :description, category = :category, "
            "tags = :tags, location_type = :location_type, ruling_power = :ruling_power, "
            "cultural_notes = :cultural_notes, updated_at = NOW() WHERE id = :wa_id"
        ),
        {
            "name": wa.name,
            "description": wa.description,
            "category": wa.category,
            "tags": wa.tags,
            "location_type": wa.location_type,
            "ruling_power": wa.ruling_power,
            "cultural_notes": Jsonb(wa.cultural_notes)
            if wa.cultural_notes is not None
            else None,
            "wa_id": wa.id,
        },
    )


async def delete_world_anvil(session: AsyncSession, wa_id: str | UUID) -> None:
    """Delete a world element from the database."""

    await session.execute(
        sa_text("DELETE FROM world_anvil WHERE id = :wa_id"), {"wa_id": wa_id}
    )


async def create_concept(session: AsyncSession, concept: Concept) -> Concept:
    """Insert ``concept`` and return it with the new ID."""

    result = await session.execute(
        sa_text(
            "INSERT INTO concept (logline, theme, conflict, protagonist, hook, "
            "genre, tone, setting, mood) VALUES (:logline, :theme, :conflict, :protagonist, :hook, :genre, :tone, :setting, :mood) "
            "RETURNING id, created_at"
        ),
        {
            "logline": concept.logline,
            "theme": concept.theme,
            "conflict": concept.conflict,
            "protagonist": concept.protagonist,
            "hook": concept.hook,
            "genre": concept.genre,
            "tone": concept.tone,
            "setting": concept.setting,
            "mood": concept.mood,
        },
    )
    row = await _maybe_await(result.fetchone())
    if row:
        concept.id, concept.created_at = row
    return concept


async def get_concept(session: AsyncSession, concept_id: str | UUID) -> Concept | None:
    """Return :class:`Concept` by ID."""

    result = await session.execute(
        sa_text(
            "SELECT id, logline, theme, conflict, protagonist, hook, genre, tone, "
            "setting, mood, created_at FROM concept WHERE id = :cid"
        ),
        {"cid": concept_id},
    )
    row = await _maybe_await(result.fetchone())
    if row is None:
        return None
    (
        cid,
        logline,
        theme,
        conflict,
        protagonist,
        hook,
        genre,
        tone,
        setting,
        mood,
        created_at,
    ) = row
    return Concept(
        id=cid,
        logline=logline,
        theme=theme,
        conflict=conflict,
        protagonist=protagonist,
        hook=hook,
        genre=genre,
        tone=tone,
        setting=setting,
        mood=mood,
        created_at=created_at,
    )


async def update_concept(session: AsyncSession, concept: Concept) -> None:
    """Persist updates to ``concept``."""

    await session.execute(
        sa_text(
            "UPDATE concept SET logline = :logline, theme = :theme, conflict = :conflict, "
            "protagonist = :protagonist, hook = :hook, genre = :genre, tone = :tone, setting = :setting, "
            "mood = :mood WHERE id = :cid"
        ),
        {
            "logline": concept.logline,
            "theme": concept.theme,
            "conflict": concept.conflict,
            "protagonist": concept.protagonist,
            "hook": concept.hook,
            "genre": concept.genre,
            "tone": concept.tone,
            "setting": concept.setting,
            "mood": concept.mood,
            "cid": concept.id,
        },
    )


async def delete_concept(session: AsyncSession, concept_id: str | UUID) -> None:
    """Delete ``concept_id`` from the database."""

    await session.execute(
        sa_text("DELETE FROM concept WHERE id = :cid"), {"cid": concept_id}
    )


async def create_story(session: AsyncSession, story: Story) -> Story:
    """Insert ``story`` and return it with the creation timestamp."""

    result = await session.execute(
        sa_text(
            "INSERT INTO story (id, title, concept_id) VALUES (:sid, :title, :cid) RETURNING created_at"
        ),
        {"sid": story.id, "title": story.title, "cid": story.concept_id},
    )
    row = await _maybe_await(result.fetchone())
    if row:
        story.created_at = row[0]
    return story


async def create_chapter(session: AsyncSession, chapter_data: Chapter) -> Chapter:
    """Insert ``chapter_data`` into the database and return it with the new ID."""

    result = await session.execute(  # type: ignore[arg-type]
        sa_text(
            "INSERT INTO chapter (title, description, order_index, structure_notes) VALUES (%s, %s, %s, %s) RETURNING id, created_at"
        ),
        (
            chapter_data.title,
            chapter_data.description,
            chapter_data.order_index,
            Jsonb(chapter_data.structure_notes)
            if chapter_data.structure_notes is not None
            else None,
        ),
    )
    row = await _maybe_await(result.fetchone())
    if row:
        chapter_data.id = row[0]
        chapter_data.created_at = row[1]
    await session.commit()
    return chapter_data


async def update_chapter(session: AsyncSession, chapter_data: Chapter) -> None:
    """Persist updates to ``chapter_data``."""

    await session.execute(
        sa_text(
            "UPDATE chapter SET title = :title, description = :description, order_index = :order_index, "
            "structure_notes = :structure_notes WHERE id = :cid"
        ),
        {
            "title": chapter_data.title,
            "description": chapter_data.description,
            "order_index": chapter_data.order_index,
            "structure_notes": Jsonb(chapter_data.structure_notes)
            if chapter_data.structure_notes is not None
            else None,
            "cid": chapter_data.id,
        },
    )


async def delete_chapter(session: AsyncSession, chapter_id: str | UUID) -> None:
    """Remove a chapter from the database."""

    await session.execute(
        sa_text("DELETE FROM chapter WHERE id = :cid"), {"cid": chapter_id}
    )


async def get_chapter(session: AsyncSession, chapter_id: str | UUID) -> Chapter | None:
    """Return the :class:`Chapter` for ``chapter_id`` or ``None`` if missing."""

    result = await session.execute(
        sa_text(
            "SELECT id, title, description, order_index, structure_notes, created_at FROM chapter WHERE id = :cid"
        ),
        {"cid": chapter_id},
    )
    row = await _maybe_await(result.fetchone())
    if row is None:
        return None
    cid, title, description, order_index, structure_notes, created_at = row
    return Chapter(
        id=cid,
        title=title,
        description=description,
        order_index=order_index,
        structure_notes=structure_notes,
        created_at=created_at,
    )


async def create_scene(session: AsyncSession, scene_data: Scene) -> Scene:
    """Insert ``scene_data`` into the database and return it with the new ID."""

    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        f"Creating scene: {scene_data.title}",
        Priority.NORMAL,
        metadata={
            "operation": "insert",
            "table": "scene",
            "scene_title": scene_data.title,
            "scene_number": scene_data.scene_number,
            "status": scene_data.status.value,
            "character_count": len(scene_data.characters) if scene_data.characters else 0,
            "has_text": bool(scene_data.text),
            "text_length": len(scene_data.text) if scene_data.text else 0
        }
    )

    try:
        result = await session.execute(
            sa_text(
                "INSERT INTO scene (title, description, text, status, "
                "scene_number, setting, characters, location_id, chapter_id) "
                "VALUES (:title, :description, :text, :status, :scene_number, :setting, :characters, :location_id, :chapter_id) RETURNING id"
            ),
            {
                "title": scene_data.title,
                "description": scene_data.description,
                "text": scene_data.text,
                "status": scene_data.status.value,
                "scene_number": scene_data.scene_number,
                "setting": scene_data.setting,
                "characters": scene_data.characters,
                "location_id": scene_data.location_id,
                "chapter_id": scene_data.chapter_id,
            },
        )
        row = await _maybe_await(result.fetchone())
        if row:
            scene_data.id = row[0]
        await session.commit()
        
        if scene_data.text:
            assert scene_data.id is not None
            await event_logger.log(
                EventType.DATABASE_OPERATION,
                f"Storing scene text for scene {scene_data.id}",
                Priority.NORMAL,
                metadata={
                    "operation": "store_text",
                    "scene_id": str(scene_data.id),
                    "text_length": len(scene_data.text)
                }
            )
            await store_scene_text_conn(session, scene_data.id, scene_data.text)
        
        duration = time.time() - start_time
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Successfully created scene: {scene_data.title} (ID: {scene_data.id})",
            Priority.NORMAL,
            metadata={
                "operation": "insert",
                "table": "scene",
                "duration": duration,
                "scene_id": str(scene_data.id),
                "scene_title": scene_data.title,
                "success": True
            }
        )
        
        return scene_data
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"Creating scene {scene_data.title}",
            metadata={
                "operation": "insert",
                "table": "scene",
                "duration": duration,
                "scene_title": scene_data.title
            }
        )
        raise


async def update_scene(session: AsyncSession, scene_data: Scene) -> None:
    """Persist updates to ``scene_data``."""

    await session.execute(
        sa_text(
            "UPDATE scene SET title = :title, description = :description, text = :text, "
            "status = :status, scene_number = :scene_number, setting = :setting, characters = :characters, "
            "location_id = :location_id, chapter_id = :chapter_id, updated_at = NOW() WHERE id = :sid"
        ),
        {
            "title": scene_data.title,
            "description": scene_data.description,
            "text": scene_data.text,
            "status": scene_data.status.value,
            "scene_number": scene_data.scene_number,
            "setting": scene_data.setting,
            "characters": scene_data.characters,
            "location_id": scene_data.location_id,
            "chapter_id": scene_data.chapter_id,
            "sid": scene_data.id,
        },
    )
    if scene_data.text:
        assert scene_data.id is not None
        await store_scene_text_conn(session, scene_data.id, scene_data.text)


async def delete_scene(session: AsyncSession, scene_id: str | UUID) -> None:
    """Delete ``scene_id`` from the database."""

    await session.execute(
        sa_text("DELETE FROM scene WHERE id = :sid"), {"sid": scene_id}
    )


async def get_scene(session: AsyncSession, scene_id: str | UUID) -> Scene | None:
    """Return the :class:`Scene` for ``scene_id`` or ``None`` if missing."""

    result = await session.execute(
        sa_text(
            "SELECT id, title, description, text, status, scene_number, "
            "setting, characters, location_id, chapter_id, created_at, updated_at "
            "FROM scene WHERE id = :sid"
        ),
        {"sid": scene_id},
    )
    row = await _maybe_await(result.fetchone())
    if row is None:
        return None
    (
        sid,
        title,
        description,
        text_value,
        status,
        scene_number,
        setting,
        characters,
        location_id,
        chapter_id,
        created_at,
        updated_at,
    ) = row
    return Scene(
        id=sid,
        title=title,
        description=description,
        text=text_value or "",
        status=SceneStatus(status),
        scene_number=scene_number,
        setting=setting,
        characters=list(characters) if characters is not None else [],
        location_id=location_id,
        chapter_id=chapter_id,
        created_at=created_at,
        updated_at=updated_at,
    )


async def update_scene_status(
    session: AsyncSession, scene_id: UUID | str, status: SceneStatus
) -> None:
    """Set ``scene_id`` to ``status``."""
    
    await session.execute(
        sa_text(
            "UPDATE scene SET status = :status, updated_at = NOW() WHERE id = :sid"
        ),
        {"status": status.value, "sid": scene_id},
    )


async def get_all_chapters(session: AsyncSession) -> list[Chapter]:
    """Return all chapters in the database."""
    
    result = await session.execute(
        sa_text(
            "SELECT id, title, description, order_index, structure_notes, created_at FROM chapter ORDER BY order_index"
        )
    )
    rows = result.fetchall()
    return [
        Chapter(
            id=r[0],
            title=r[1],
            description=r[2],
            order_index=r[3],
            structure_notes=r[4],
            created_at=r[5],
        )
        for r in rows
    ]


async def get_all_characters(session: AsyncSession) -> list[CharacterProfile]:
    """Return all character profiles in the database."""
    
    result = await session.execute(
        sa_text(
            "SELECT id, name, full_name, aliases, gender, age, birth_date, death_date, species, role, rank, backstory, beliefs, desires, intentions, motivations, fatal_flaw, created_at, updated_at FROM character_profile ORDER BY name"
        )
    )
    rows = result.fetchall()
    return [
        CharacterProfile(
            id=r[0],
            name=r[1],
            full_name=r[2],
            aliases=list(r[3]) if r[3] is not None else [],
            gender=r[4],
            age=r[5],
            birth_date=r[6],
            death_date=r[7],
            species=r[8],
            role=r[9],
            rank=r[10],
            backstory=r[11],
            beliefs=r[12],
            desires=r[13],
            intentions=r[14],
            motivations=r[15],
            fatal_flaw=r[16],
            created_at=r[17],
            updated_at=r[18],
        )
        for r in rows
    ]


async def get_scenes_by_status(
    session: AsyncSession, status: SceneStatus
) -> list[Scene]:
    """Return scenes filtered by ``status``."""

    result = await session.execute(
        sa_text(
            "SELECT id, title, description, text, status, scene_number, "
            "setting, characters, location_id, chapter_id, created_at, updated_at "
            "FROM scene WHERE status = :status ORDER BY created_at"
        ),
        {"status": status.value},
    )
    rows = result.fetchall()
    return [
        Scene(
            id=r[0],
            title=r[1],
            description=r[2],
            text=r[3] or "",
            status=SceneStatus(r[4]),
            scene_number=r[5],
            setting=r[6],
            characters=list(r[7]) if r[7] is not None else [],
            location_id=r[8],
            chapter_id=r[9],
            created_at=r[10],
            updated_at=r[11],
        )
        for r in rows
    ]


async def get_all_scenes(session: AsyncSession) -> list[Scene]:
    """Return all scenes in the database."""
    
    result = await session.execute(
        sa_text(
            "SELECT id, title, description, text, status, scene_number, "
            "setting, characters, location_id, chapter_id, created_at, updated_at "
            "FROM scene ORDER BY created_at"
        )
    )
    rows = result.fetchall()
    return [
        Scene(
            id=r[0],
            title=r[1],
            description=r[2],
            text=r[3] or "",
            status=SceneStatus(r[4]),
            scene_number=r[5],
            setting=r[6],
            characters=list(r[7]) if r[7] is not None else [],
            location_id=r[8],
            chapter_id=r[9],
            created_at=r[10],
            updated_at=r[11],
        )
        for r in rows
    ]


__all__ = [
    "record_story_feedback",
    "set_configuration_conn",
    "get_all_characters",
    "set_configuration",
    "get_configuration",
    "load_configuration_file",
    "create_approval_log",
    "create_task_log",
    "create_story_feedback",
    "get_feedback_for_scene",
    "create_character_profile",
    "get_character_profile",
    "update_character_profile",
    "delete_character_profile",
    "create_character_relationship",
    "create_world_anvil",
    "get_world_anvil",
    "update_world_anvil",
    "delete_world_anvil",
    "create_concept",
    "get_concept",
    "update_concept",
    "delete_concept",
    "create_story",
    "create_chapter",
    "get_chapter",
    "update_chapter",
    "delete_chapter",
    "create_scene",
    "get_scene",
    "update_scene",
    "update_scene_status",
    "delete_scene",
    "get_scenes_by_status",
    "get_all_scenes",
    "get_all_chapters",
]
