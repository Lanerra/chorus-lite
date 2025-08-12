# src/chorus/agents/integration_manager.py
"""IntegrationManager agent that handles story integration, validation, and finalization."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text as sa_text

from chorus.agents.base import Agent
from chorus.canon.postgres import _maybe_await, get_pg
from chorus.core.llm import call_llm_structured
from chorus.core.logs import log_message
from chorus.models import (
    Chapter,
    Scene,
    SceneStatus,
    Story,
)
from chorus.models.responses import (
    StoryFeedback,
    ValidationResult,
)


class IntegrationManager(Agent):
    """Agent responsible for story integration, validation, and finalization."""

    def __init__(self, *, model: str | None = None) -> None:
        """Initialize the IntegrationManager agent.

        Parameters
        ----------
        model:
            Optional override for the default LLM model used by the IntegrationManager.
            If provided, this value will be used instead of the
            ``INTEGRATION_MANAGER_MODEL`` environment variable when invoking language
            models.
        """
        super().__init__(model=model, default_model_env="INTEGRATION_MANAGER_MODEL")

    # --- Story Integration Methods ---

    async def integrate_scenes(
        self, scenes: list[Scene], chapter: Chapter | None = None
    ) -> Chapter:
        """Integrate a collection of scenes into a cohesive chapter."""
        if not scenes:
            raise ValueError("Cannot integrate empty scene list")

        # Sort scenes by their intended order
        sorted_scenes = sorted(scenes, key=lambda s: getattr(s, "order_index", 0))

        # Generate chapter content
        scene_summaries = [
            f"Scene {i+1}: {scene.title} - {scene.summary or scene.description}"
            for i, scene in enumerate(sorted_scenes)
        ]

        prompt = (
            "Create a cohesive chapter by integrating these scenes. "
            "Ensure smooth transitions between scenes, consistent tone, "
            "and logical narrative flow. Provide a chapter title and summary.\n"
            "Scenes to integrate:\n" + "\n".join(scene_summaries) + "\n"
            "Scene contents:\n"
            + "\n".join(
                [
                    f"=== {scene.title} ===\n{scene.text}"
                    for scene in sorted_scenes
                    if scene.text
                ]
            )
            + "\n"
            "Respond only with JSON that matches the Chapter schema. DO NOT include any commentary or additional text."
        )

        integrated_chapter = await call_llm_structured(self.model, prompt, Chapter)

        # Preserve existing chapter metadata if provided
        if chapter:
            integrated_chapter.id = chapter.id
            integrated_chapter.order_index = chapter.order_index

        # Link scenes to chapter
        integrated_chapter.scenes = sorted_scenes

        return integrated_chapter

    async def validate_story_structure(self, story: Story) -> ValidationResult:
        """Validate overall story structure for completeness and coherence."""
        issues = []
        warnings = []

        # Check for required story elements
        if not story.title:
            issues.append("Story is missing a title")

        if not story.chapters:
            issues.append("Story has no chapters")
        elif len(story.chapters) == 0:
            issues.append("Story contains empty chapters list")

        # Validate chapters
        for i, chapter in enumerate(story.chapters or []):
            if not chapter.title:
                issues.append(f"Chapter {i+1} is missing a title")

            if not chapter.scenes:
                warnings.append(f"Chapter {i+1} ({chapter.title}) has no scenes")
            elif len(chapter.scenes) == 0:
                warnings.append(
                    f"Chapter {i+1} ({chapter.title}) has empty scenes list"
                )

            # Validate scenes within chapter
            for j, scene in enumerate(chapter.scenes or []):
                if not scene.title:
                    issues.append(f"Chapter {i+1}, Scene {j+1} is missing a title")

                if not scene.text:
                    issues.append(
                        f"Chapter {i+1}, Scene {j+1} ({scene.title}) has no content"
                    )

                if scene.status == SceneStatus.QUEUED:
                    warnings.append(
                        f"Chapter {i+1}, Scene {j+1} ({scene.title}) is still queued"
                    )
                elif scene.status == SceneStatus.DRAFT:
                    warnings.append(
                        f"Chapter {i+1}, Scene {j+1} ({scene.title}) is still in draft"
                    )

        # Use LLM for deeper structural analysis
        prompt = (
            "Analyze this story structure for narrative coherence, pacing, and completeness. "
            "Identify any structural issues, plot holes, or inconsistencies.\n"
            f"Story: {story.model_dump_json()}\n"
            "Respond only with JSON that matches the ValidationResult schema. DO NOT include any commentary or additional text."
        )

        llm_validation = await call_llm_structured(self.model, prompt, ValidationResult)

        # Combine manual checks with LLM analysis
        all_issues = issues + (llm_validation.errors or [])
        all_warnings = warnings + (llm_validation.warnings or [])

        return ValidationResult(
            is_valid=len(all_issues) == 0,
            errors=all_issues,
            warnings=all_warnings,
            suggestions=llm_validation.suggestions or [],
        )

    async def generate_story_summary(self, story: Story) -> str:
        """Generate a comprehensive summary of the complete story."""
        prompt = (
            "Create a comprehensive summary of this story. Include the main plot, "
            "character arcs, themes, and key events. The summary should capture "
            "the essence of the story for readers or publishers.\n"
            f"Story: {story.model_dump_json()}\n"
            "Respond with a well-written summary text, not JSON."
        )

        return await self.call_llm(prompt)

    async def create_story_outline(self, story: Story) -> dict[str, Any]:
        """Create a detailed outline of the story structure."""
        outline = {
            "title": story.title,
            "summary": await self.generate_story_summary(story),
            "chapters": [],
            "total_scenes": 0,
            "estimated_word_count": 0,
        }

        for chapter in story.chapters or []:
            chapter_info = {
                "title": chapter.title,
                "description": chapter.description,
                "scene_count": len(chapter.scenes or []),
                "scenes": [],
            }

            word_count = 0
            for scene in chapter.scenes or []:
                scene_info = {
                    "title": scene.title,
                    "description": scene.description,
                    "characters": scene.characters or [],
                    "status": scene.status.value if scene.status else "unknown",
                    "word_count": len((scene.text or "").split()),
                }
                chapter_info["scenes"].append(scene_info)
                word_count += scene_info["word_count"]

            chapter_info["word_count"] = word_count
            outline["chapters"].append(chapter_info)
            outline["total_scenes"] += chapter_info["scene_count"]
            outline["estimated_word_count"] += word_count

        return outline

    # --- Quality Assurance Methods ---

    async def review_story_quality(self, story: Story) -> StoryFeedback:
        """Conduct comprehensive quality review of the complete story."""
        prompt = (
            "Conduct a comprehensive quality review of this story. Evaluate: "
            "1. Plot coherence and pacing "
            "2. Character development and consistency "
            "3. Writing quality and style "
            "4. Dialogue effectiveness "
            "5. Theme development "
            "6. Overall reader engagement "
            "Provide detailed feedback and suggestions for improvement.\n"
            f"Story: {story.model_dump_json()}\n"
            "Respond only with JSON that matches the StoryFeedback schema. DO NOT include any commentary or additional text."
        )

        return await call_llm_structured(self.model, prompt, StoryFeedback)

    async def check_consistency_across_chapters(
        self, story: Story
    ) -> dict[str, list[str]]:
        """Check for consistency issues across all chapters in the story."""
        consistency_issues: dict[str, list[str]] = {
            "character_inconsistencies": [],
            "timeline_issues": [],
            "world_building_conflicts": [],
            "tone_variations": [],
        }

        if not story.chapters or len(story.chapters) < 2:
            return consistency_issues

        # Analyze character consistency across chapters
        all_characters = set()
        for chapter in story.chapters:
            for scene in chapter.scenes or []:
                all_characters.update(scene.characters or [])

        for character in all_characters:
            prompt = (
                f"Analyze {character}'s consistency across all chapters. "
                "Look for contradictions in personality, abilities, knowledge, or behavior.\n"
                f"Story: {story.model_dump_json()}\n"
                "Return a JSON array of inconsistency descriptions, or empty array if consistent."
            )

            try:
                result = await self.call_llm(prompt)
                issues = json.loads(result)
                if isinstance(issues, list) and issues:
                    consistency_issues["character_inconsistencies"].extend(
                        [f"{character}: {issue}" for issue in issues]
                    )
            except Exception as e:
                await self.log_message(
                    f"Error checking character consistency for {character}: {e}"
                )

        # Check timeline consistency
        prompt = (
            "Analyze the timeline and sequence of events across all chapters. "
            "Identify any chronological inconsistencies or impossible event sequences.\n"
            f"Story: {story.model_dump_json()}\n"
            "Return a JSON array of timeline issue descriptions, or empty array if consistent."
        )

        try:
            result = await self.call_llm(prompt)
            issues = json.loads(result)
            if isinstance(issues, list):
                consistency_issues["timeline_issues"] = issues
        except Exception as e:
            await self.log_message(f"Error checking timeline consistency: {e}")

        return consistency_issues

    async def validate_scene_transitions(self, story: Story) -> list[str]:
        """Validate transitions between scenes and chapters."""
        transition_issues = []

        for chapter_idx, chapter in enumerate(story.chapters or []):
            scenes = chapter.scenes or []

            # Check transitions within chapter
            for scene_idx in range(len(scenes) - 1):
                current_scene = scenes[scene_idx]
                next_scene = scenes[scene_idx + 1]

                prompt = (
                    "Analyze the transition between these two consecutive scenes. "
                    "Identify any jarring transitions, logical gaps, or flow issues.\n"
                    f"Scene 1: {current_scene.model_dump_json()}\n"
                    f"Scene 2: {next_scene.model_dump_json()}\n"
                    "Return a brief description of any transition issues, or 'SMOOTH' if the transition works well."
                )

                try:
                    result = await self.call_llm(prompt)
                    if result.strip() != "SMOOTH":
                        transition_issues.append(
                            f"Chapter {chapter_idx + 1}, Scene {scene_idx + 1} to {scene_idx + 2}: {result}"
                        )
                except Exception as e:
                    await self.log_message(f"Error checking scene transition: {e}")

            # Check transition to next chapter
            if chapter_idx < len(story.chapters) - 1:
                next_chapter = story.chapters[chapter_idx + 1]

                if scenes and next_chapter.scenes:
                    last_scene = scenes[-1]
                    first_next_scene = next_chapter.scenes[0]

                    prompt = (
                        "Analyze the transition between the last scene of one chapter "
                        "and the first scene of the next chapter.\n"
                        f"Last scene of Chapter {chapter_idx + 1}: {last_scene.model_dump_json()}\n"
                        f"First scene of Chapter {chapter_idx + 2}: {first_next_scene.model_dump_json()}\n"
                        "Return a brief description of any transition issues, or 'SMOOTH' if the transition works well."
                    )

                    try:
                        result = await self.call_llm(prompt)
                        if result.strip() != "SMOOTH":
                            transition_issues.append(
                                f"Chapter {chapter_idx + 1} to {chapter_idx + 2}: {result}"
                            )
                    except Exception as e:
                        await self.log_message(
                            f"Error checking chapter transition: {e}"
                        )

        return transition_issues

    # --- Finalization Methods ---

    async def finalize_story(
        self, story: Story, perform_validation: bool = True
    ) -> Story:
        """Perform final processing and validation of the complete story."""
        if perform_validation:
            # Validate story structure
            validation = await self.validate_story_structure(story)
            if not validation.is_valid:
                error_summary = "; ".join(validation.errors or [])
                raise ValueError(f"Story validation failed: {error_summary}")

        # Generate final summary
        story.summary = await self.generate_story_summary(story)

        # Update story status to indicate completion
        # This would be implemented based on the actual Story model
        # story.status = StoryStatus.COMPLETE

        # Persist final story to database
        await self.persist_story(story)

        return story

    async def generate_export_formats(self, story: Story) -> dict[str, str]:
        """Generate story in various export formats."""
        formats = {}

        # Plain text format
        text_parts = [f"# {story.title}\n"]
        if story.summary:
            text_parts.append(f"{story.summary}\n")

        for chapter in story.chapters or []:
            text_parts.append(f"\n## Chapter: {chapter.title}\n")
            if chapter.description:
                text_parts.append(f"{chapter.description}\n")

            for scene in chapter.scenes or []:
                text_parts.append(f"\n### {scene.title}\n")
                if scene.text:
                    text_parts.append(f"{scene.text}\n")

        formats["text"] = "\n".join(text_parts)

        # Markdown format (similar but with proper markdown formatting)
        formats["markdown"] = formats[
            "text"
        ]  # Could be enhanced with better formatting

        # JSON format
        formats["json"] = story.model_dump_json(indent=2)

        return formats

    async def create_publication_package(self, story: Story) -> dict[str, Any]:
        """Create a comprehensive package for publication or sharing."""
        # Validate story first
        validation = await self.validate_story_structure(story)
        quality_review = await self.review_story_quality(story)
        consistency_check = await self.check_consistency_across_chapters(story)

        # Generate exports
        exports = await self.generate_export_formats(story)

        # Create outline
        outline = await self.create_story_outline(story)

        return {
            "story": story.model_dump(),
            "outline": outline,
            "validation": validation.model_dump(),
            "quality_review": quality_review.model_dump(),
            "consistency_analysis": consistency_check,
            "exports": exports,
            "metadata": {
                "total_chapters": len(story.chapters or []),
                "total_scenes": sum(
                    len(ch.scenes or []) for ch in (story.chapters or [])
                ),
                "estimated_word_count": sum(
                    len((scene.text or "").split())
                    for chapter in (story.chapters or [])
                    for scene in (chapter.scenes or [])
                ),
                "completion_status": "ready_for_publication"
                if validation.is_valid
                else "needs_revision",
            },
        }

    # --- Helper Methods ---

    async def log_message(self, message: str) -> None:
        """Log a message using the logging system."""
        await log_message(message)

    async def call_llm(self, prompt: str) -> str:
        """Call the LLM with a simple prompt and return text response."""
        from chorus.core.llm import call_llm

        return await call_llm(self.model, prompt)

    async def persist_story(self, story: Story) -> Story:
        """Persist the complete story to database."""
        async with get_pg() as conn:
            if story.id:
                # Update existing story
                await conn.execute(
                    sa_text(
                        "UPDATE story SET title = :title, summary = :summary, "
                        "updated_at = CURRENT_TIMESTAMP WHERE id = :id"
                    ),
                    {
                        "id": story.id,
                        "title": story.title,
                        "summary": story.summary,
                    },
                )
            else:
                # Create new story
                result = await conn.execute(
                    sa_text(
                        "INSERT INTO story (title, summary, created_at, updated_at) "
                        "VALUES (:title, :summary, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                        "RETURNING id"
                    ),
                    {
                        "title": story.title,
                        "summary": story.summary,
                    },
                )
                row = await _maybe_await(result.fetchone())
                if row:
                    story.id = row[0]

            await conn.commit()

        return story

    async def get_story_by_id(self, story_id: str) -> Story | None:
        """Retrieve a complete story by ID from the database."""
        async with get_pg() as conn:
            # Get story
            story_result = await conn.execute(
                sa_text("SELECT id, title, summary FROM story WHERE id = :id"),
                {"id": story_id},
            )
            story_row = await _maybe_await(story_result.fetchone())

            if not story_row:
                return None

            story_id_db, title, summary = story_row

            # Get chapters
            chapters_result = await conn.execute(
                sa_text(
                    "SELECT id, title, description, order_index FROM chapter "
                    "WHERE story_id = :story_id ORDER BY order_index"
                ),
                {"story_id": story_id_db},
            )
            chapter_rows = await _maybe_await(chapters_result.fetchall())

            chapters = []
            for ch_row in chapter_rows:
                ch_id, ch_title, ch_desc, ch_order = ch_row

                # Get scenes for this chapter
                scenes_result = await conn.execute(
                    sa_text(
                        "SELECT id, title, description, text, status, characters, summary "
                        "FROM scene WHERE chapter_id = :chapter_id ORDER BY id"
                    ),
                    {"chapter_id": ch_id},
                )
                scene_rows = await _maybe_await(scenes_result.fetchall())

                scenes = []
                for sc_row in scene_rows:
                    (
                        sc_id,
                        sc_title,
                        sc_desc,
                        sc_text,
                        sc_status,
                        sc_chars,
                        sc_summary,
                    ) = sc_row

                    scene = Scene(
                        id=sc_id,
                        title=sc_title,
                        description=sc_desc,
                        text=sc_text,
                        status=SceneStatus(sc_status)
                        if sc_status
                        else SceneStatus.DRAFT,
                        characters=sc_chars or [],
                        summary=sc_summary,
                    )
                    scenes.append(scene)

                chapter = Chapter(
                    id=ch_id,
                    title=ch_title,
                    description=ch_desc,
                    order_index=ch_order,
                    scenes=scenes,
                )
                chapters.append(chapter)

            story = Story(
                id=story_id_db, title=title, summary=summary, chapters=chapters
            )

            return story
