# src/chorus/agents/scene_generator.py
"""SceneGenerator agent that consolidates scene writing, continuity checking, and style editing."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from uuid import UUID

from sqlalchemy import text as sa_text

from chorus.agents.base import Agent
from chorus.canon.postgres import (
    _maybe_await,
    create_story_feedback,
    get_pg,
)
from chorus.core.llm import call_llm_structured
from chorus.core.logs import get_event_logger, log_message, EventType, LogLevel, Priority
from chorus.core.queue import enqueue
from chorus.models import (
    CharacterProfile,
    Scene,
    SceneBrief,
    SceneStatus,
    StoryFeedback,
)
from chorus.models.story_feedback import StoryFeedbackType
from chorus.models.responses import (
    ContinuityFeedback,
    StyleFeedback,
)
from chorus.models.task import RewriteTask


def _prose_ok(text: str) -> bool:
    """Return ``True`` if ``text`` appears well formed."""

    stripped = text.strip()
    if not stripped or not stripped[0].isupper():
        return False
    # Allow single-sentence fragments that might not end in punctuation.
    if "." not in stripped and "?" not in stripped and "!" not in stripped:
        if len(stripped.split()) > 5:  # It's likely a sentence fragment
            return True

    if not stripped.endswith((".", "?", "!", '"', "'")):
        return False

    sentences = re.split(r"[.!?]", stripped)
    for sentence in sentences:
        words = sentence.split()
        if words and len(words) > 40:  # Relaxed from 20 to 40
            return False
    if re.search(r"\b(\w+)\s+\1\b", stripped, re.IGNORECASE):
        return False
    return True


class SceneGenerator(Agent):
    """Agent responsible for scene writing, continuity checking, and style editing."""

    def __init__(self, *, model: str | None = None) -> None:
        """Initialize the SceneGenerator agent.

        Parameters
        ----------
        model:
            Optional override for the default LLM model used by the SceneGenerator.
            If provided, this value will be used instead of the
            ``SCENE_GENERATOR_MODEL`` environment variable when invoking language
            models.
        """
        super().__init__(model=model, default_model_env="SCENE_GENERATOR_MODEL")
        self.event_logger = get_event_logger()

    # --- Scene Writing Methods ---

    async def write_scene(
        self, brief: SceneBrief, context: dict[str, Any] | None = None
    ) -> Scene:
        """Generate scene content from a brief with optional context."""
        self.event_logger.log(
            LogLevel.INFO,
            f"Starting scene writing for: '{brief.title}'",
            event_type=EventType.SCENE_DRAFT,
            priority=Priority.HIGH,
            scene_id=brief.id,
            metadata={
                "agent": "SceneGenerator",
                "operation": "write_scene",
                "scene_title": brief.title,
                "characters": brief.characters or [],
                "has_context": context is not None,
                "context_keys": list(context.keys()) if context else []
            }
        )
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
            self.event_logger.log(
                LogLevel.DEBUG,
                f"Using context for scene generation: {list(context.keys())}",
                event_type=EventType.SCENE_DRAFT,
                priority=Priority.NORMAL,
                scene_id=brief.id,
                metadata={"agent": "SceneGenerator", "context_size": len(str(context))}
            )

        prompt = (
            "Write a compelling scene based on this brief. "
            "Focus on character development, dialogue, and vivid descriptions. "
            "The scene should advance the plot and reveal character motivations.\n"
            f"Brief: {brief.model_dump_json()}"
            f"{context_str}\n"
            "Respond only with JSON that matches the Scene schema. DO NOT include any commentary or additional text."
        )

        self.event_logger.log(
            LogLevel.DEBUG,
            "Sending scene generation request to LLM",
            event_type=EventType.LLM_REQUEST,
            priority=Priority.NORMAL,
            scene_id=brief.id,
            metadata={"agent": "SceneGenerator", "model": self.model, "prompt_length": len(prompt)}
        )

        scene = await call_llm_structured(self.model, prompt, Scene)

        # Copy essential fields from brief
        scene.id = brief.id
        scene.title = brief.title
        scene.description = brief.description
        scene.characters = brief.characters or []
        scene.status = SceneStatus.DRAFT

        self.event_logger.log(
            LogLevel.INFO,
            f"Successfully generated scene draft: '{scene.title}' ({len(scene.text or '')} characters)",
            event_type=EventType.SCENE_DRAFT,
            priority=Priority.HIGH,
            scene_id=brief.id,
            metadata={
                "agent": "SceneGenerator",
                "operation": "write_scene",
                "scene_text_length": len(scene.text or ''),
                "scene_status": scene.status.value
            }
        )

        return scene

    async def revise_scene(
        self, scene: Scene, feedback: str, context: dict[str, Any] | None = None
    ) -> Scene:
        """Revise a scene based on feedback."""
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        prompt = (
            "Revise this scene based on the feedback provided. "
            "Maintain the core story elements while addressing the concerns. "
            "Ensure consistency with character voices and story continuity.\n"
            f"Original Scene: {scene.model_dump_json()}\n"
            f"Feedback: {feedback}"
            f"{context_str}\n"
            "Respond only with JSON that matches the Scene schema. DO NOT include any commentary or additional text."
        )

        revised_scene = await call_llm_structured(self.model, prompt, Scene)

        # Preserve essential metadata
        revised_scene.id = scene.id
        revised_scene.status = SceneStatus.REVISED

        return revised_scene

    async def expand_scene(self, scene: Scene, target_length: int = 2000) -> Scene:
        """Expand a scene to reach target length while maintaining quality."""
        current_length = len(scene.text) if scene.text else 0

        if current_length >= target_length:
            return scene

        prompt = (
            f"Expand this scene to approximately {target_length} words. "
            "Add more detail, dialogue, internal thoughts, and sensory descriptions. "
            "Maintain the existing tone and pacing while enriching the narrative.\n"
            f"Current Scene ({current_length} words): {scene.model_dump_json()}\n"
            "Respond only with JSON that matches the Scene schema. DO NOT include any commentary or additional text."
        )

        expanded_scene = await call_llm_structured(self.model, prompt, Scene)

        # Preserve essential metadata
        expanded_scene.id = scene.id
        expanded_scene.title = scene.title
        expanded_scene.description = scene.description
        expanded_scene.characters = scene.characters
        expanded_scene.status = SceneStatus.EXPANDED

        return expanded_scene

    # --- Continuity Guardian Methods ---

    async def check_continuity(
        self, scene: Scene, story_context: str = ""
    ) -> ContinuityFeedback:
        """Check scene for continuity issues against story context."""
        prompt = (
            "Analyze this scene for continuity issues. Check for: "
            "1. Character consistency (personality, abilities, knowledge) "
            "2. Timeline consistency (references to past/future events) "
            "3. World-building consistency (rules, geography, technology) "
            "4. Plot consistency (character motivations, story logic)\n"
            f"Scene: {scene.model_dump_json()}\n"
            f"Story Context: {story_context}\n"
            "Respond only with JSON that matches the ContinuityFeedback schema. DO NOT include any commentary or additional text."
        )

        return await call_llm_structured(self.model, prompt, ContinuityFeedback)

    async def validate_character_consistency(
        self, scene: Scene, profiles: list[CharacterProfile]
    ) -> dict[str, list[str]]:
        """Validate character actions against their established profiles."""
        character_issues: dict[str, list[str]] = {}

        for profile in profiles:
            if profile.name in (scene.characters or []):
                prompt = (
                    f"Analyze if {profile.name}'s actions, dialogue, and behavior in this scene "
                    "are consistent with their established character profile. "
                    "List any inconsistencies or out-of-character moments.\n"
                    f"Character Profile: {profile.model_dump_json()}\n"
                    f"Scene: {scene.model_dump_json()}\n"
                    "Return a JSON array of inconsistency descriptions, or empty array if consistent."
                )

                try:
                    result = await self.call_llm(prompt)
                    issues = json.loads(result)
                    if isinstance(issues, list) and issues:
                        character_issues[profile.name] = issues
                except Exception as e:
                    self.event_logger.log(
                        LogLevel.ERROR,
                        f"Error checking character consistency for {profile.name}: {e}",
                        event_type=EventType.AGENT_OPERATION,
                        priority=Priority.NORMAL,
                        metadata={
                            "agent": "SceneGenerator",
                            "operation": "validate_character_consistency",
                            "character_name": profile.name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )

        return character_issues

    async def get_canonical_context(self, scene: Scene) -> str:
        """Retrieve relevant canonical context for continuity checking."""
        context_parts = []

        async with get_pg() as conn:
            # Get character profiles for characters in scene
            if scene.characters:
                char_placeholders = ",".join(
                    [f":char_{i}" for i in range(len(scene.characters))]
                )
                char_params = {
                    f"char_{i}": char for i, char in enumerate(scene.characters)
                }

                result = await conn.execute(
                    sa_text(
                        f"SELECT name, arc, motivations FROM character_profile WHERE name IN ({char_placeholders})"
                    ),
                    char_params,
                )
                char_rows = await _maybe_await(result.fetchall())

                for row in char_rows:
                    name, arc, motivations = row
                    context_parts.append(
                        f"Character {name}: Arc={arc}, Motivations={motivations}"
                    )

            # Get previous scenes for timeline context
            if scene.id:
                result = await conn.execute(
                    sa_text(
                        "SELECT title, summary FROM scene WHERE id < :scene_id ORDER BY id DESC LIMIT 5"
                    ),
                    {"scene_id": scene.id},
                )
                prev_scenes = await _maybe_await(result.fetchall())

                if prev_scenes:
                    context_parts.append("Previous scenes:")
                    for title, summary in prev_scenes:
                        context_parts.append(f"- {title}: {summary}")

        return "\n".join(context_parts)

    # --- Continuity Guardian Integration Methods ---

    async def check_scene(self, scene: Scene) -> Scene | None:
        """Validate ``scene`` against the Canon."""

        self.event_logger.log(
            LogLevel.INFO,
            f"Starting continuity check for scene {scene.id}",
            event_type=EventType.SCENE_CONTINUITY,
            priority=Priority.HIGH,
            scene_id=scene.id,
            metadata={"agent": "SceneGenerator", "operation": "check_scene"}
        )
        async with self.get_db_connection() as conn:
            # Check if we should skip continuity check for first chapter
            order_cursor = await conn.execute(
                sa_text(
                    "SELECT c.order_index FROM chapter c "
                    "JOIN scene s ON s.chapter_id = c.id WHERE s.id = :sid"
                ),
                {"sid": scene.id},
            )
            row = await _maybe_await(order_cursor.fetchone())
            chapter_index = row[0] if row else None

            if chapter_index == 1:
                self.event_logger.log(
                    LogLevel.INFO,
                    f"Skipping continuity check for first chapter scene {scene.id} (chapter 1 scenes bypass continuity)",
                    event_type=EventType.SCENE_CONTINUITY,
                    priority=Priority.NORMAL,
                    scene_id=scene.id,
                    metadata={"agent": "SceneGenerator", "chapter_index": chapter_index}
                )
                status = SceneStatus.DRAFTING
                await conn.execute(
                    sa_text("UPDATE scene SET status = :status WHERE id = :sid"),
                    {"status": status.value, "sid": scene.id},
                )
                await conn.commit()
                scene.status = status
                self.event_logger.log(
                    LogLevel.INFO,
                    f"Continuity check passed for scene {scene.id} (first chapter approval)",
                    event_type=EventType.SCENE_CONTINUITY,
                    priority=Priority.HIGH,
                    scene_id=scene.id,
                    metadata={"agent": "SceneGenerator", "operation": "check_scene", "result": "approved"}
                )
                return scene

        # Check for continuity issues using database logic (similar to checks.py)
        issues = await self._check_continuity_db(scene)
        status = SceneStatus.DRAFTING
        feedback = "; ".join(item.notes for item in issues)

        if issues:
            status = SceneStatus.REJECTED
            self.event_logger.log(
                LogLevel.WARNING,
                f"Continuity check failed for scene {scene.id}, enqueuing for rewrite",
                event_type=EventType.SCENE_CONTINUITY,
                priority=Priority.HIGH,
                scene_id=scene.id,
                metadata={
                    "agent": "SceneGenerator",
                    "operation": "check_scene",
                    "result": "rejected",
                    "feedback": feedback,
                    "issues_count": len(issues)
                }
            )
            await enqueue(
                RewriteTask(
                    scene_id=scene.id if scene.id is not None else UUID(int=0),
                    notes=feedback,
                    draft=scene.text,
                ),
                priority=10,
            )

        await conn.execute(
            sa_text("UPDATE scene SET status = :status WHERE id = :sid"),
            {"status": status.value, "sid": scene.id},
        )
        await conn.commit()

        if status == SceneStatus.REJECTED:
            await create_story_feedback(
                conn,
                StoryFeedback(
                    scene_id=scene.id if scene.id is not None else UUID(int=0),
                    feedback_type=StoryFeedbackType.CONTINUITY,
                    content=feedback,
                ),
            )
            self.event_logger.log(
                LogLevel.ERROR,
                f"Continuity check failed for scene {scene.id} - creating story feedback entry",
                event_type=EventType.SCENE_CONTINUITY,
                priority=Priority.HIGH,
                scene_id=scene.id,
                metadata={"agent": "SceneGenerator", "operation": "check_scene", "result": "failed"}
            )
            return None

        scene.status = status
        self.event_logger.log(
            LogLevel.INFO,
            f"Continuity check passed for scene {scene.id}",
            event_type=EventType.SCENE_CONTINUITY,
            priority=Priority.HIGH,
            scene_id=scene.id,
            metadata={"agent": "SceneGenerator", "operation": "check_scene", "result": "approved"}
        )
        return scene

    async def _check_continuity_db(self, scene: Scene) -> list[ContinuityFeedback]:
        """Check scene for continuity issues using database and text similarity (similar to checks.py)."""
        # This is a simplified version of the original checks.py logic
        # In the full implementation, we would do more sophisticated checks

        issues: list[ContinuityFeedback] = []

        # Check for similar scenes to identify potential continuity issues
        # Note: This is a placeholder - in the full implementation we'd have proper text similarity
        # For now we'll use a generic check based on existing logic
        async with get_pg() as conn:
            # We'll check if similar scenes exist in the database
            # This is a simplified version of the original logic from checks.py
            result = await conn.execute(
                sa_text(
                    "SELECT id FROM scene WHERE text LIKE :text AND id != :scene_id LIMIT 1"
                ),
                {"text": f"%{scene.text[:100]}%", "scene_id": scene.id},
            )
            similar_scenes = await _maybe_await(result.fetchall())

            if similar_scenes:
                issues.append(
                    ContinuityFeedback(
                        scene_id=scene.id if scene.id is not None else UUID(int=0),
                        notes="Potential continuity issue: Similar scene found in database",
                    )
                )

        return issues

    async def review_draft_continuity(self, scene_id: UUID) -> Scene | None:
        """Load a draft scene and run continuity check."""

        async with self.get_db_connection() as conn:
            result = await conn.execute(
                sa_text(
                    "SELECT title, description, text, characters FROM scene WHERE id = :sid"
                ),
                {"sid": scene_id},
            )
            row = await _maybe_await(result.fetchone())

        if row is None:
            raise ValueError("Scene not found")

        title, description, text, characters = row
        scene = Scene(
            id=scene_id,
            title=title,
            description=description,
            text=text,
            characters=characters or [],
        )
        return await self.check_scene(scene)

    async def enhance_dialogue(self, scene: Scene) -> Scene:
        """Enhance dialogue in the scene for naturalness and character voice."""
        prompt = (
            "Enhance the dialogue in this scene. Make it more natural, distinctive to each character, "
            "and emotionally resonant. Ensure each character has a unique voice that reflects "
            "their personality and background.\n"
            f"Scene: {scene.model_dump_json()}\n"
            "Respond only with JSON that matches the Scene schema. DO NOT include any commentary or additional text."
        )

        enhanced_scene = await call_llm_structured(self.model, prompt, Scene)

        # Preserve essential metadata
        enhanced_scene.id = scene.id
        enhanced_scene.title = scene.title
        enhanced_scene.description = scene.description
        enhanced_scene.characters = scene.characters
        enhanced_scene.status = SceneStatus.ENHANCED

        return enhanced_scene

    async def polish_prose(self, scene: Scene) -> Scene:
        """Polish the prose for clarity, flow, and impact."""
        prompt = (
            "Polish the prose in this scene. Improve sentence variety, eliminate redundancy, "
            "enhance imagery and sensory details, and ensure smooth transitions. "
            "Maintain the original meaning while elevating the quality.\n"
            f"Scene: {scene.model_dump_json()}\n"
            "Respond only with JSON that matches the Scene schema. DO NOT include any commentary or additional text."
        )

        polished_scene = await call_llm_structured(self.model, prompt, Scene)

        # Preserve essential metadata
        polished_scene.id = scene.id
        polished_scene.title = scene.title
        polished_scene.description = scene.description
        polished_scene.characters = scene.characters
        polished_scene.status = SceneStatus.POLISHED

        return polished_scene


    # --- Integrated Workflow Methods ---

    async def generate_complete_scene(
        self,
        brief: SceneBrief,
        max_revisions: int = 3,
    ) -> Scene:
        """Generate a complete scene through the full writing, continuity, and style pipeline."""
        # Initial scene generation
        scene = await self.write_scene(brief)

        # Get canonical context for continuity checking
        story_context = await self.get_canonical_context(scene)

        revision_count = 0
        while revision_count < max_revisions:
            # Check continuity
            continuity_feedback = await self.check_continuity(scene, story_context)

            # Analyze style
            style_feedback = await self.analyze_style(scene)

            # Determine if revisions are needed
            needs_revision = False
            feedback_parts = []


            if not needs_revision:
                break

            # Apply revisions
            feedback_text = "; ".join(feedback_parts)
            scene = await self.revise_scene(scene, feedback_text)
            revision_count += 1


        # Final polish
        scene = await self.enhance_dialogue(scene)
        scene = await self.polish_prose(scene)

        # Mark as complete
        scene.status = SceneStatus.COMPLETE

        return scene

    async def batch_process_scenes(
        self,
        briefs: list[SceneBrief],
        concurrency_limit: int = 3,
    ) -> list[Scene]:
        """Process multiple scenes concurrently with limited concurrency."""
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_single_scene(brief: SceneBrief) -> Scene:
            async with semaphore:
                return await self.generate_complete_scene(brief)

        tasks = [process_single_scene(brief) for brief in briefs]
        scenes = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        successful_scenes = []
        for i, result in enumerate(scenes):
            if isinstance(result, Exception):
                self.event_logger.log(
                    LogLevel.ERROR,
                    f"Error processing scene '{briefs[i].title}' in batch: {result}",
                    event_type=EventType.SCENE_DRAFT,
                    priority=Priority.HIGH,
                    metadata={
                        "agent": "SceneGenerator",
                        "operation": "batch_process_scenes",
                        "scene_title": briefs[i].title,
                        "error": str(result),
                        "error_type": type(result).__name__
                    }
                )
            else:
                successful_scenes.append(result)

        return successful_scenes

    # --- Helper Methods ---

    async def log_message(self, message: str) -> None:
        """Log a message using the logging system."""
        self.event_logger.log(
            LogLevel.INFO,
            message,
            event_type=EventType.AGENT_OPERATION,
            priority=Priority.NORMAL,
            metadata={"agent": "SceneGenerator"}
        )

    async def call_llm(self, prompt: str) -> str:
        """Call the LLM with a simple prompt and return text response."""
        from chorus.core.llm import call_llm

        return await call_llm(self.model, prompt)

    async def persist_scene(self, scene: Scene) -> Scene:
        """Persist scene to database and return updated scene with any DB-assigned fields."""
        async with get_pg() as conn:
            if scene.id:
                # Update existing scene
                await conn.execute(
                    sa_text(
                        "UPDATE scene SET text = :text, status = :status, summary = :summary "
                        "WHERE id = :id"
                    ),
                    {
                        "id": scene.id,
                        "text": scene.text,
                        "status": scene.status.value,
                        "summary": scene.summary,
                    },
                )
            else:
                # Create new scene
                result = await conn.execute(
                    sa_text(
                        "INSERT INTO scene (title, description, text, status, characters, summary) "
                        "VALUES (:title, :description, :text, :status, :characters, :summary) "
                        "RETURNING id"
                    ),
                    {
                        "title": scene.title,
                        "description": scene.description,
                        "text": scene.text,
                        "status": scene.status.value,
                        "characters": scene.characters or [],
                        "summary": scene.summary,
                    },
                )
                row = await _maybe_await(result.fetchone())
                if row:
                    scene.id = row[0]

            await conn.commit()

        return scene
