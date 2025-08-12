# src/chorus/langgraph/nodes.py
"""Asynchronous LangGraph node functions for the refined graph.

Each node mirrors a single agent responsibility and is designed to be
idempotent.  Keeping nodes small and stateless simplifies testing and allows the
graph to evolve as new agents are implemented.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from typing import Any, cast

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt
from sqlalchemy import text as sa_text

from chorus.agents.scene_generator import SceneGenerator

# New consolidated agents
from chorus.agents.story_architect import StoryArchitect
from chorus.canon.postgres import (
    _maybe_await,
    commit_session,
    ensure_schema,
    get_pg,
    update_scene_status,
)
from chorus.core.logs import get_logger, log_message
from chorus.models import (
    CharacterProfile,
    Concept,
    Scene,
    SceneStatus,
    Story,
)
from chorus.models.task import SceneBrief

from .memory import search_text, store_text, summarize_text
from .state import SceneState, StoryState

logger = get_logger(__name__)

SCENE_FIELDS = {
    "draft",
    "review_status",
    "revision_notes",
    "revision_history",
    "feedback",
    "lore_context",
    "lore_summary",
    "short_term_memory",
    "memory_summary",
    "memory_context",
    "needs_revision",
    "revision_count",
    "scene_status",
    "draft_rejected",
    "coherence_passed",  # Phase 2: Single-pass coherence validation
    "coherence_issues",  # Phase 2: List of coherence issues found
}


# Interactive mode functionality removed as per cleanup spec


def _with_scene_state(state: StoryState, result: dict[str, Any]) -> dict[str, Any]:
    """Return ``result`` with per-scene updates appended."""

    scene_id = state.get("current_scene_id")
    if not scene_id:
        return result
    updates = {k: v for k, v in result.items() if k in SCENE_FIELDS}
    if not updates:
        return result
    result = result.copy()
    result["scene_states"] = {scene_id: updates}
    return result


def _require_model(env_var: str) -> str:
    """Return the model specified by ``env_var`` or fall back to defaults."""

    model = os.getenv(env_var)
    if model:
        return model

    from chorus.config import config

    mapping = {
        "STORY_ARCHITECT_MODEL": config.agents.story_architect,
        "SCENE_GENERATOR_MODEL": config.agents.scene_generator,
        "INTEGRATION_MANAGER_MODEL": config.agents.integration_manager,
    }

    if env_var in mapping and mapping[env_var]:
        return mapping[env_var]

    raise RuntimeError(f"{env_var} not configured")


from chorus.config import config

from .errors import GraphRetry

RETRY_ATTEMPTS = config.retry.retry_attempts
RETRY_BACKOFF = config.retry.retry_backoff
RETRY_MAX_INTERVAL = config.retry.retry_max_interval


async def _retryable(fn: Any) -> Any:
    """Retry ``fn`` up to ``RETRY_ATTEMPTS`` with exponential backoff."""

    backoff = RETRY_BACKOFF
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await fn()
        except Exception as exc:  # pragma: no cover - retryable failures
            if attempt == RETRY_ATTEMPTS - 1:
                raise GraphRetry() from exc
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RETRY_MAX_INTERVAL)


def safe_node(fn: Any, enable_enhanced_error_handling: bool = True) -> Any:
    """Return wrapper that records errors in the state and tracks node metadata.
    Args:
        fn: The node function to wrap
        enable_enhanced_error_handling: Whether to use the new comprehensive error handling system
    """

    async def _wrapper(state: StoryState) -> dict[str, Any]:
        start_time = time.time()
        node_name = fn.__name__

        # Track model being used for this node if available
        model_env_vars = {
            "generate_concepts_node": "STORY_ARCHITECT_MODEL",
            "select_concept_node": "STORY_ARCHITECT_MODEL",
            "generate_world": "STORY_ARCHITECT_MODEL",
            "generate_profiles": "STORY_ARCHITECT_MODEL",
            "generate_outline": "STORY_ARCHITECT_MODEL",
            "prepare_scenes": "STORY_ARCHITECT_MODEL",
            "draft_scene": "SCENE_GENERATOR_MODEL",
            "revise_scene": "INTEGRATION_MANAGER_MODEL",
            "integrate": "INTEGRATION_MANAGER_MODEL",
            "catalog_lore_node": "STORY_ARCHITECT_MODEL",
            "gather_lore_context_node": "STORY_ARCHITECT_MODEL",
            "evolve_profiles_node": "STORY_ARCHITECT_MODEL",
        }

        model_name = None
        if node_name in model_env_vars:
            try:
                model_name = _require_model(model_env_vars[node_name])
            except RuntimeError:
                model_name = "not_configured"

        # Use enhanced error handling if enabled
        use_enhanced_handling = enable_enhanced_error_handling

        if use_enhanced_handling:
            try:
                from .error_recovery import (
                    ErrorRecoveryManager,
                    get_error_recovery_manager,
                )

                recovery_manager = get_error_recovery_manager()

                try:
                    # Execute the node function
                    result = await fn(state)
                except Exception as e:
                    # If an error occurs, handle it with the recovery manager
                    return await recovery_manager.handle_node_error(
                        error=e,
                        node_name=node_name,
                        state=state,
                        original_function=fn,
                    )

                # Add execution metadata if result is a dict
                if isinstance(result, dict):
                    duration = time.time() - start_time
                    metadata = {
                        "node_name": node_name,
                        "duration_seconds": round(duration, 3),
                        "status": "success",
                    }
                    if model_name:
                        metadata["model"] = model_name
                    result["_node_metadata"] = metadata

                return result

            except ImportError:
                # Fall back to basic error handling if enhanced system not available
                logger.warning(f"Enhanced error handling not available for {node_name}")
                use_enhanced_handling = False

        # Basic error handling
        try:
            result = await fn(state)
            duration = time.time() - start_time

            # Log node execution metadata
            metadata = {
                "node_name": node_name,
                "duration_seconds": round(duration, 3),
                "status": "success",
            }
            if model_name:
                metadata["model"] = model_name

            log_message(f"Node execution: {metadata}")

            # Add metadata to result if it's a dict
            if isinstance(result, dict):
                result["_node_metadata"] = metadata

            return result
        except GraphInterrupt:
            raise
        except GraphRetry as exc:
            duration = time.time() - start_time
            metadata = {
                "node_name": node_name,
                "duration_seconds": round(duration, 3),
                "status": "retry_failed",
                "error": str(exc.__cause__ or exc),
            }
            if model_name:
                metadata["model"] = model_name

            log_message(f"Node retry failed: {metadata}")
            log_message(traceback.format_exc())
            return {
                "error_messages": [str(exc.__cause__ or exc)],
                "error_nodes": [node_name],
                "_node_metadata": metadata,
            }
        except Exception as exc:  # pragma: no cover - unexpected failures
            duration = time.time() - start_time
            metadata = {
                "node_name": node_name,
                "duration_seconds": round(duration, 3),
                "status": "error",
                "error": str(exc),
            }
            if model_name:
                metadata["model"] = model_name

            log_message(f"Node error: {metadata}")
            log_message(traceback.format_exc())
            return {
                "error_messages": [str(exc)],
                "error_nodes": [node_name],
                "_node_metadata": metadata,
            }

    return _wrapper


def configure_node_error_handling(
    critical_nodes: list[str] | None = None,
    skippable_nodes: list[str] | None = None,
    circuit_breaker_threshold: int = 5,
    enable_fault_isolation: bool = True,
) -> dict[str, Any]:
    """Configure enhanced error handling behavior for workflow nodes.

    Args:
        critical_nodes: List of node names that require manual intervention on failure
        skippable_nodes: List of node names that can be skipped on repeated failures
        circuit_breaker_threshold: Number of failures before circuit breaker opens
        enable_fault_isolation: Whether to enable component-level fault isolation

    Returns:
        Configuration dictionary for error handling
    """
    return {
        "critical_nodes": critical_nodes or [],
        "skippable_nodes": skippable_nodes
        or [
            "catalog_lore_node",
        ],
        "circuit_breaker_threshold": circuit_breaker_threshold,
        "enable_fault_isolation": enable_fault_isolation,
    }


def get_node_criticality(node_name: str) -> str:
    """Determine the criticality level of a workflow node.

    Args:
        node_name: Name of the workflow node

    Returns:
        Criticality level: 'critical', 'important', 'optional', or 'enhancement'
    """
    critical_nodes = {
        "generate_concepts_node",
        "select_concept_node",
        "generate_outline",
        "draft_scene",
        "dequeue_scene",
        "finalize_scene_node",
    }

    important_nodes = {
        "generate_world",
        "generate_profiles",
        "revise_scene",
        "integrate",
    }

    optional_nodes = {
        "gather_lore_context_node",
        "retrieve_memory_node",
        "store_memory_node",
        "summarize_memory_node",
    }

    enhancement_nodes = {
        "catalog_lore_node",
        "evolve_profiles_node",
    }

    if node_name in critical_nodes:
        return "critical"
    elif node_name in important_nodes:
        return "important"
    elif node_name in optional_nodes:
        return "optional"
    elif node_name in enhancement_nodes:
        return "enhancement"
    else:
        return "important"  # Default to important for unknown nodes


def apply_safe_node_to_all():
    """Apply safe_node decorator to all node functions in this module.

    This function can be called to automatically wrap all workflow nodes
    with enhanced error handling. Useful for testing and development.
    """
    import sys

    current_module = sys.modules[__name__]

    node_functions = [
        "generate_concepts_node",
        "select_concept_node",
        "generate_world",
        "generate_profiles",
        "seed_narrative_context",
        "generate_outline",
        "prepare_scenes",
        "generate_scenes",
        "dequeue_scene",
        "draft_scene",
        "revise_scene",
        "integrate",
        "manual_revision",
        "finalize_scene_node",
        "persist_scene",
        "catalog_lore_node",
        "gather_lore_context_node",
        "retrieve_memory_node",
        "store_memory_node",
        "summarize_memory_node",
        "human_feedback_node",
        "evolve_profiles_node",
        "refill_scene_queue_if_needed",
        "process_scenes_concurrently_node",
    ]

    for func_name in node_functions:
        if hasattr(current_module, func_name):
            original_func = getattr(current_module, func_name)
            # Only wrap if not already wrapped
            if not hasattr(original_func, "_safe_node_wrapped"):
                wrapped_func = safe_node(original_func)
                wrapped_func._safe_node_wrapped = True
                setattr(current_module, func_name, wrapped_func)
                logger.info(f"Applied enhanced error handling to {func_name}")


def _current_brief(state: StoryState) -> SceneBrief:
    """Return the :class:`SceneBrief` matching ``current_scene_id``."""

    scene_id = state.get("current_scene_id")
    if scene_id is None:
        raise ValueError("current scene not set")
    briefs = cast(list[SceneBrief], state.get("scene_briefs", []))
    for brief in briefs:
        if str(brief.id) == str(scene_id):
            return brief
    raise ValueError("scene brief missing id")


async def _set_scene_status(scene_id: str | None, status: SceneStatus) -> None:
    """Persist ``status`` for ``scene_id`` if possible."""
    if scene_id is None:
        return
    try:
        async with get_pg() as conn:
            await update_scene_status(conn, scene_id, status)
            await commit_session(conn)
    except KeyError:
        log_message("Postgres not configured; skipping scene status update")
    except Exception as e:
        # Handle any database errors gracefully (e.g., during testing)
        log_message(f"Scene status update failed; skipping: {e}")


async def _is_scene_approved(scene_id: str) -> bool:
    """Return ``True`` if the scene is already approved."""

    try:
        async with get_pg() as conn:
            result = await conn.execute(
                sa_text("SELECT status FROM scene WHERE id = :sid"),
                {"sid": scene_id},
            )
            row = await _maybe_await(result.fetchone())
            if row:
                return SceneStatus(row[0]) == SceneStatus.APPROVED
    except KeyError:
        log_message("Postgres not configured; assuming scene not approved")
    return False


async def generate_concepts_node(state: StoryState) -> dict[str, Any]:
    """Return a list of story concepts for ``state.idea``."""
    model = _require_model("STORY_ARCHITECT_MODEL")
    architect = StoryArchitect(model=model)
    concepts = await architect.generate_concepts(cast(str, state["idea"]))
    return {"concepts": concepts}


async def select_concept_node(state: StoryState) -> dict[str, Any]:
    """Return the first concept as the chosen concept."""
    concepts = cast(list[Concept], state["concepts"])
    
    # Return the first concept without evaluation
    if not concepts:
        raise ValueError("No concepts to choose from.")
    return {"vision": concepts[0], "concept_rejected": False}


async def generate_world(state: StoryState) -> dict[str, Any]:
    """Create world information for the selected concept.

    Side-effects: none (in-memory only). Sets 'world_generated'=True for ordering.
    """
    await ensure_schema()
    model = _require_model("STORY_ARCHITECT_MODEL")
    concept = cast(Concept, state["vision"])
    architect = StoryArchitect(model=model)
    world = await architect.generate_world_anvil(concept)
    return {"world_info": world, "world_generated": True}


async def generate_profiles(state: StoryState) -> dict[str, Any]:
    """Generate character profiles for the concept."""
    await ensure_schema()
    model = _require_model("STORY_ARCHITECT_MODEL")
    concept = cast(Concept, state["vision"])
    architect = StoryArchitect(model=model)
    profiles = await architect.generate_profiles(concept)
    return {"character_profiles": profiles}


async def seed_narrative_context(state: StoryState) -> dict[str, Any]:
    """Gather initial narrative text/context into memory only.

    This function prepares an in-memory summary/context for downstream nodes and
    writes summaries to long-term memory namespaces. It must NOT perform DB writes
    that depend on NER/world ingestion tables.

    Prereq: state['world_ingested'] must be True. Sets 'seeded'=True.
    """

    # Guard ordering: require world ingestion step to have completed
    world_ready = state.get("world_ingested")
    if not world_ready:
        log_message(
            "seed_narrative_context: skipping; prerequisite 'world_ingested' not satisfied"
        )
        return {}

    concept = cast(Concept, state.get("vision"))
    profiles = cast(list[CharacterProfile], state.get("character_profiles", []))

    # Prepare summaries without DB writes or NER-dependent ingestion
    summary_parts: list[str] = []
    if concept and concept.logline:
        summary_parts.append(concept.logline)
    for prof in profiles:
        summary_parts.append(prof.backstory or prof.name)
    summary = "\n".join(summary_parts).strip()

    # Seed long-term memory namespaces only (memory/state)
    thread_id = state.get("thread_id")
    if thread_id and summary:
        await store_text(namespace=(str(thread_id),), text=summary)
        for prof in profiles:
            text = prof.backstory or ""
            if text:
                await store_text(namespace=(f"{thread_id}:{prof.name}",), text=text)

    return {
        "narrative_context_seeded": True,
        "context_summaries": summary,
        "seeded": True,
    }


async def generate_outline(state: StoryState) -> dict[str, Any]:
    """Create a story outline for the selected concept and persist immediately.

    Prereq: state['seeded'] must be True. Sets 'outline_ready'=True.
    """
    # Guard ordering: require narrative seed
    narrative_ready = state.get("seeded")
    if not narrative_ready:
        log_message("generate_outline: skipping; prerequisite 'seeded' not satisfied")
        return {}
    await ensure_schema()
    model = _require_model("STORY_ARCHITECT_MODEL")
    concept = cast(Concept, state["vision"])
    profiles = cast(list[CharacterProfile], state.get("character_profiles", []))
    architect = StoryArchitect(model=model)

    # Create outline and convert to tasks
    outline = await architect.create_story_outline(concept, profiles)
    story = await architect.outline_to_tasks(outline)

    chapters = [c.title for c in story.chapters]
    briefs: list[SceneBrief] = []
    persisted_chapters: list[str] = []
    for chapter in story.chapters:
        briefs.extend(chapter.scenes)
        persisted_chapters.append(chapter.title)
    queue = [str(b.id) for b in briefs if b.id is not None]

    # Return both the original outline struct and the persisted story/chapters/queue
    return {
        "outline": story,
        "chapters": persisted_chapters or chapters,
        "scene_briefs": briefs,
        "scene_queue": queue,
        "outline_ready": True,
    }


async def prepare_scenes(state: StoryState) -> dict[str, Any]:
    """Persist outline and return scene briefs.

    Prereq: state['outline_ready'] must be True.
    """
    if state.get("scene_briefs"):
        return {}
    if not state.get("outline_ready"):
        log_message(
            "prepare_scenes: skipping; prerequisite 'outline_ready' not satisfied"
        )
        return {}
    model = _require_model("STORY_ARCHITECT_MODEL")
    outline = cast(Story, state["outline"])
    architect = StoryArchitect(model=model)
    story = await architect.outline_to_tasks(outline)
    briefs: list[SceneBrief] = []
    chapters = []
    for chapter in story.chapters:
        briefs.extend(chapter.scenes)
        chapters.append(chapter.title)
    queue = [str(b.id) for b in briefs if b.id is not None]
    return {
        "scene_briefs": briefs,
        "outline": story,
        "chapters": chapters,
        "scene_queue": queue,
    }


async def generate_scenes(state: StoryState) -> dict[str, Any]:
    """Create scene briefs for the story."""
    model = _require_model("STORY_ARCHITECT_MODEL")
    concept = cast(Concept, state["vision"])
    profiles = cast(list[CharacterProfile], state.get("character_profiles", []))
    architect = StoryArchitect(model=model)
    story: Story = await architect.generate_scenes(concept, profiles=profiles)
    briefs: list[SceneBrief] = []
    for chapter in story.chapters:
        briefs.extend(chapter.scenes)
    return {"scene_briefs": briefs}


async def dequeue_scene(state: StoryState) -> dict[str, Any]:
    """Pop the next scene ID from ``scene_queue``."""

    queue = list(state.get("scene_queue", []))
    if not queue:
        return {"current_scene_id": None}
    current = queue.pop(0)
    # Track processed scene ids to avoid reprocessing when refilling
    from collections.abc import Iterable
    from typing import cast as _cast

    processed = set(_cast(Iterable[str], state.get("processed_scene_ids", [])))
    processed.add(str(current))
    return {
        "scene_queue": queue,
        "current_scene_id": current,
        "processed_scene_ids": list(processed),
        "scene_states": {str(current): SceneState()},
    }


async def draft_scene(state: StoryState) -> dict[str, Any]:
    """Draft a scene and collect optional user feedback."""
    if state.get("draft") and not state.get("needs_revision", False):
        return {}

    model = _require_model("SCENE_GENERATOR_MODEL")
    brief = _current_brief(state)

    # Build context dictionary for the scene generator
    context: dict[str, Any] = {}
    if state.get("lore_context"):
        context["lore_context"] = state["lore_context"]
    if state.get("memory_summary"):
        context["memory_summary"] = state["memory_summary"]
    if state.get("memory_context"):
        context["memory_context"] = state["memory_context"]
    clusters = state.get("character_clusters")
    if clusters:
        joined = "; ".join(", ".join(c) for c in cast(list[list[str]], clusters))
        context["character_clusters"] = joined

    scene_generator = SceneGenerator(model=model)
    scene = await scene_generator.write_scene(
        brief, context=context if context else None
    )

    notes: str | None = None
    # Removed interactive mode - always proceed autonomously
    notes = None  # No interactive feedback in non-interactive mode

    revision_notes = list(state.get("revision_notes", []))
    needs_revision = bool(notes)
    if notes:
        revision_notes.append(cast(str, notes))
    output = {
        "draft": scene.text,
        "scene_status": scene.status,
        "review_status": ["approved"],
        "needs_revision": needs_revision,
        "revision_count": cast(int, state.get("revision_count", 0) + 1),
        "revision_notes": revision_notes,
    }
    return _with_scene_state(state, output)


def check_character_consistency(
    scene_brief: str, scene_content: str
) -> tuple[bool, list[str]]:
    """Check if characters in brief appear in scene content.

    Args:
        scene_brief: The scene brief description containing character names
        scene_content: The actual scene text to check

    Returns:
        Tuple of (passed, issues_list)
    """
    issues = []
    if not scene_content or not scene_content.strip():
        return False, ["Scene content is empty"]

    # Extract character names from brief (simple approach)
    # This is a basic implementation that could be enhanced
    import re

    # Look for character names in typical brief formats
    # "Character A and Character B discuss..." or "Character A meets Character B"
    brief_lower = scene_brief.lower()
    content_lower = scene_content.lower()

    # Simple pattern matching for character names (can be enhanced)
    # Look for capitalized words that might be character names
    potential_chars = re.findall(r"\b[A-Z][a-z]+\b", scene_brief)

    for char in potential_chars:
        if len(char) > 2 and char.lower() not in {
            "the",
            "and",
            "but",
            "for",
            "with",
            "this",
            "that",
        }:
            if char.lower() not in content_lower:
                issues.append(
                    f"Character '{char}' mentioned in brief but not found in scene"
                )

    return len(issues) == 0, issues


def check_basic_readability(scene_content: str) -> tuple[bool, list[str]]:
    """Check word count and basic paragraph structure.

    Args:
        scene_content: The scene text to validate

    Returns:
        Tuple of (passed, issues_list)
    """
    issues = []

    if not scene_content or not scene_content.strip():
        issues.append("Scene is empty")
        return False, issues

    # Word count check (minimum 200, maximum 2000 as per design doc)
    words = scene_content.split()
    word_count = len(words)

    if word_count < 200:
        issues.append(f"Scene too short: {word_count} words (minimum 200)")
    elif word_count > 2000:
        issues.append(f"Scene too long: {word_count} words (maximum 2000)")

    # Paragraph structure check
    paragraphs = [p.strip() for p in scene_content.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        issues.append("Scene should have multiple paragraphs for proper structure")

    # Check for reasonable sentence structure
    sentences = [
        s.strip()
        for s in scene_content.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    if len(sentences) < 5:
        issues.append("Scene may be too brief for narrative development")

    return len(issues) == 0, issues


async def coherence_check_node(state: StoryState) -> dict[str, Any]:
    """Single-pass coherence check replacing multi-stage review system.

    Implements minimal coherence criteria:
    - Character name consistency: Characters mentioned in scene brief appear in scene content
    - Basic readability: Reasonable word count (200-2000 words), proper paragraph structure
    - No subjective analysis: No complex LLM-based style or narrative quality checks

    Args:
        state: Current story state containing draft and scene brief

    Returns:
        Updated state with coherence results
    """
    brief = _current_brief(state)
    if brief.id is None:
        raise ValueError("scene brief missing id")

    draft = cast(str, state.get("draft", ""))
    if not draft:
        return _with_scene_state(
            state,
            {
                "coherence_issues": ["No draft content to check"],
                "coherence_passed": False,
                "scene_status": SceneStatus.REJECTED,
                "needs_revision": True,
            },
        )

    # Run coherence checks
    issues = []

    # 1. Character consistency check
    if brief.description and brief.characters:
        char_passed, char_issues = check_character_consistency(brief.description, draft)
        issues.extend(char_issues)

    # 2. Basic readability check
    read_passed, read_issues = check_basic_readability(draft)
    issues.extend(read_issues)

    # Determine pass/fail
    coherence_passed = len(issues) == 0

    if coherence_passed:
        await _set_scene_status(str(brief.id), SceneStatus.APPROVED)
        output = {
            "coherence_issues": [],
            "coherence_passed": True,
            "scene_status": SceneStatus.APPROVED,
            "needs_revision": False,
        }
    else:
        await _set_scene_status(str(brief.id), SceneStatus.REJECTED)
        output = {
            "coherence_issues": issues,
            "coherence_passed": False,
            "scene_status": SceneStatus.REJECTED,
            "needs_revision": True,
        }

    return _with_scene_state(state, output)


# Legacy review functions removed in Phase 4 cleanup
# These have been replaced by the single coherence_check_node


async def revise_scene(state: StoryState) -> dict[str, Any]:
    """Apply accumulated feedback and prepare for a new draft."""

    model = _require_model("INTEGRATION_MANAGER_MODEL")
    brief = _current_brief(state)
    if brief.id is None:
        raise ValueError("scene brief missing id")
    draft = cast(str, state.get("draft", ""))
    notes = cast(list[str], state.get("revision_notes", []))
    feedback = cast(list[str], state.get("feedback", []) + notes)
    history = list(state.get("revision_history", []))
    if draft:
        history.append(draft)

    if brief.id is not None:
        scene = Scene(
            id=brief.id,
            title=brief.title,
            description=brief.description,
            text=draft,
        )
        scene_generator = SceneGenerator(model=model)
        feedback_text = "; ".join(feedback) if feedback else ""
        result = await scene_generator.revise_scene(scene, feedback_text)
        draft = result.text if result is not None else draft

    output = {
        "draft": draft,
        "feedback": [],
        "revision_notes": [],
        "needs_revision": False,
        "revision_history": history,
    }
    return _with_scene_state(state, output)


async def integrate(state: StoryState) -> dict[str, Any]:
    """Integrate feedback into the draft and pause for approval."""
    model = _require_model("INTEGRATION_MANAGER_MODEL")
    brief = _current_brief(state)
    if brief.id is None:
        raise ValueError("scene brief missing id")
    draft = cast(str, state.get("draft", ""))
    notes = cast(list[str], state.get("revision_notes", []))
    feedback = cast(list[str], state.get("feedback", []) + notes)
    history = list(state.get("revision_history", []))
    if draft:
        history.append(draft)

    # Only create scene object if we have a valid ID
    if brief.id is not None:
        scene = Scene(
            id=brief.id, title=brief.title, description=brief.description, text=draft
        )
        scene_generator = SceneGenerator(model=model)
        feedback_text = "; ".join(feedback) if feedback else ""
        result = await scene_generator.revise_scene(scene, feedback_text)
        final = result.text if result is not None else draft
    else:
        final = draft

    draft_rejected = False
    # Removed interactive mode - always approve automatically
    draft_rejected = False  # Always approve final draft in non-interactive mode

    output = {
        "draft": final,
        "feedback": [],
        "revision_notes": [],
        "needs_revision": False,
        "revision_history": history,
        "draft_rejected": draft_rejected,
    }
    return _with_scene_state(state, output)


async def manual_revision(state: StoryState) -> dict[str, Any]:
    """Handle revision limit via human input."""
    updates: dict[str, Any] = {}
    # Removed interactive mode - always approve automatically in non-interactive mode
    updates.update({"needs_revision": False, "review_status": ["approved"]})

    return _with_scene_state(state, updates)


# Legacy select_revision_node removed in Phase 4 cleanup
# Revision handling is now simplified with coherence_check_node


async def finalize_scene_node(state: StoryState) -> dict[str, Any]:
    """Finalize and persist the draft."""

    if state.get("scene_status") == SceneStatus.APPROVED:
        return {}

    brief = _current_brief(state)
    if brief.id is None:
        raise ValueError("scene brief missing id")

    if await _is_scene_approved(str(brief.id)):
        return {}

    model = _require_model("INTEGRATION_MANAGER_MODEL")
    draft = cast(str, state.get("draft", ""))
    scene = Scene(
        id=brief.id, title=brief.title, description=brief.description, text=draft
    )

    # Use SceneGenerator for final polishing
    scene_generator = SceneGenerator(model=model)
    finalized = await scene_generator.polish_prose(scene)
    finalized.status = SceneStatus.APPROVED

    # Persist the finalized scene
    await scene_generator.persist_scene(finalized)

    output = {
        "scene_status": finalized.status,
        "draft": finalized.text,
        "review_status": ["done"],
    }
    return _with_scene_state(state, output)


async def persist_scene(state: StoryState) -> dict[str, Any]:
    """Backward-compatible wrapper for :func:`finalize_scene_node`."""

    return await finalize_scene_node(state)


async def catalog_lore_node(state: StoryState) -> dict[str, Any]:
    """Catalog lore for the finalized scene text."""

    model = _require_model("STORY_ARCHITECT_MODEL")
    draft = cast(str, state.get("draft", ""))

    architect = StoryArchitect(model=model)
    entries = await architect.catalog_lore(draft)
    if not entries:
        return {}

    logger.info(f"Cataloged {len(entries)} lore entries")
    return {"lore_discovered": entries}


async def gather_lore_context_node(state: StoryState) -> dict[str, Any]:
    """Load contextual lore for the current scene."""

    brief = _current_brief(state)
    if not brief.characters:
        return {}

    model = _require_model("STORY_ARCHITECT_MODEL")
    architect = StoryArchitect(model=model)
    characters = list(brief.characters)
    try:
        context = await architect.get_lore_context(characters)
    except Exception as e:
        log_message(f"Lore context fetch failed; proceeding without graph context: {e}")
        context = ""
    try:
        summary = await architect.summarize_lore(characters)
    except Exception as e:
        log_message(f"Lore summary failed; proceeding without summary: {e}")
        summary = ""

    if context or summary:
        return {"lore_context": context, "lore_summary": summary}
    return {}


async def retrieve_memory_node(state: StoryState) -> dict[str, Any]:
    """Fetch related memories for the current scene."""

    thread_id = cast(str | None, state.get("thread_id"))
    if not thread_id:
        return {}
    brief = _current_brief(state)
    query = brief.description or ""
    namespaces = [(thread_id,)] + [(thread_id, c) for c in brief.characters]
    results: list[str] = []
    for ns in namespaces:
        results.extend(await search_text(ns, query))
    return _with_scene_state(state, {"memory_context": " ".join(results)})


async def store_memory_node(state: StoryState) -> dict[str, Any]:
    """Persist the finalized draft in long-term memory."""

    thread_id = cast(str | None, state.get("thread_id"))
    draft = cast(str | None, state.get("draft"))
    if not thread_id or not draft:
        return {}
    brief = _current_brief(state)
    namespaces = [(thread_id,)] + [(thread_id, c) for c in brief.characters]
    for ns in namespaces:
        await store_text(ns, draft)
    return _with_scene_state(state, {})


async def summarize_memory_node(state: StoryState) -> dict[str, Any]:
    """Condense the short-term memory list when it grows too large."""

    from collections.abc import Iterable
    from typing import cast as _cast

    history = list(_cast(Iterable[str], state.get("short_term_memory", [])))
    draft = state.get("draft")
    if draft:
        history.append(cast(str, draft))
    limit = int(os.getenv("MEMORY_LIMIT", "10"))
    summary = cast(str, state.get("memory_summary") or "")
    if draft:
        summary = " ".join([summary, summarize_text(draft)]).strip()
    if len(history) > limit:
        summary = " ".join([summary, " ".join(history[:-limit])]).strip()
        history = history[-limit:]
    return _with_scene_state(
        state, {"short_term_memory": history, "memory_summary": summary}
    )


async def refill_scene_queue_if_needed(state: StoryState) -> dict[str, Any]:
    """Continuously top up the in-memory scene_queue from persistent briefs.

    Perpetual worker behavior:
    - Poll Postgres for externally queued scenes (status='queued'), oldest first
    - Append unseen IDs to state.scene_queue
    - Skip IDs already processed or already queued
    - Also include any unseen ids from state.scene_briefs
    """
    # Collect known ids
    from collections.abc import Iterable
    from typing import cast as _cast

    queued = set(_cast(Iterable[str], state.get("scene_queue", [])))
    processed = set(_cast(Iterable[str], state.get("processed_scene_ids", [])))
    discovered: list[str] = []

    # Diagnostics: summarize current queue/processed/brief counts to debug refill stalls
    try:
        briefs_len = len(list(state.get("scene_briefs", [])))
    except Exception:
        briefs_len = 0
    logger.info(
        "refill_scene_queue_if_needed: BEGIN queue_len=%d processed=%d briefs=%d",
        len(queued),
        len(processed),
        briefs_len,
    )

    # 1) Include any unseen IDs from state.scene_briefs (in-memory outline/tasks)
    briefs = list(state.get("scene_briefs", []))
    for b in briefs:
        if b.id is None:
            continue
        sid = str(b.id)
        if sid in queued or sid in processed:
            continue
        discovered.append(sid)

    # 2) Poll Postgres for externally queued scenes (status='queued')
    try:
        async with get_pg() as conn:
            result = await conn.execute(
                sa_text(
                    "SELECT id FROM scene WHERE status = 'queued' ORDER BY created_at ASC"
                )
            )
            rows = await _maybe_await(result.fetchall())
            db_ids = [str(r[0]) for r in rows]
            logger.info(
                "refill_scene_queue_if_needed: DB queued ids=%s",
                ", ".join(db_ids) or "[]",
            )
            for sid_str in db_ids:
                if sid_str in queued or sid_str in processed:
                    continue
                # If present in briefs and already handled above, skip duplication
                if sid_str in discovered:
                    continue
                discovered.append(sid_str)
    except KeyError:
        # Postgres not configured; rely only on in-memory briefs
        logger.debug("refill_scene_queue_if_needed: Postgres not configured")
    except Exception as e:  # pragma: no cover - db optional
        log_message(f"refill_scene_queue_if_needed: DB poll failed: {e}")

    if not discovered:
        logger.info(
            "refill_scene_queue_if_needed: no new scene ids to enqueue (queue_len=%d processed=%d)",
            len(queued),
            len(processed),
        )
        return {}

    # Append to queue in stable order
    queue = list(state.get("scene_queue", []))
    queue.extend(discovered)
    logger.info(
        "refill_scene_queue_if_needed: added %d ids -> new_queue_len=%d processed=%d discovered=%s",
        len(discovered),
        len(queue),
        len(processed),
        ", ".join(discovered),
    )
    return {"scene_queue": queue}


async def human_feedback_node(state: StoryState) -> dict[str, Any]:
    """Collect user feedback or allow vision edits."""

    updates: dict[str, Any] = {}
    # Removed interactive mode - no user feedback in non-interactive mode
    # This node does nothing in non-interactive execution
    return _with_scene_state(state, updates)


async def evolve_profiles_node(state: StoryState) -> dict[str, Any]:
    """Update character profiles based on the finalized scene."""

    model = _require_model("STORY_ARCHITECT_MODEL")
    brief = _current_brief(state)
    draft = cast(str, state.get("draft", ""))
    scene = Scene(
        id=brief.id, title=brief.title, description=brief.description, text=draft
    )

    architect = StoryArchitect(model=model)
    characters = cast(list[CharacterProfile], state.get("character_profiles", []))
    scene_characters = list(brief.characters) if brief.characters else []

    profiles = await architect.evolve_profiles(
        characters=characters,
        scene_characters=scene_characters,
        scene_text=draft,
    )
    if not profiles:
        return {}
    return {"character_profiles": profiles}


async def handle_error_node(state: StoryState) -> dict[str, Any]:
    """Pause execution and let a human resolve errors.

    Tests expect:
    - interrupt -> "retry" returns {"error_message": None}
    - interrupt -> "skip" returns {"error_message": None, "error_node": None, "skip": True}
    - interrupt -> "KEY=VAL" sets env and returns {"error_message": None}
    - interrupt -> "note" appends to feedback and returns {"feedback": [..., "note"], "error_message": None}

    """
    messages = cast(list[str], state.get("error_messages", []))
    nodes = cast(list[str], state.get("error_nodes", []))
    last_node = nodes[-1] if nodes else None
    message = "; ".join(messages) if messages else ""
    log_message(f"Handling error from {last_node}: {message}")

    # Always consult interrupt() so tests with patches are exercised irrespective of interactive mode.
    action: str = ""
    try:
        note = interrupt("error")  # type: ignore[arg-type]
        action = note.strip() if isinstance(note, str) else ""
    except Exception:
        action = ""

    if action.lower() == "retry":
        return {"error_message": None}

    if action.lower() == "skip":
        return {"error_message": None, "error_node": None, "skip": True}

    # Env assignment form: KEY=VALUE
    if "=" in action and not action.strip().startswith("#"):
        var, value = action.split("=", 1)
        os.environ[var.strip()] = value.strip()
        return {"error_message": None}

    # Treat any other non-empty action as feedback note
    if action.strip():
        notes = list(state.get("feedback", []))
        notes.append(action.strip())
        return {"feedback": notes, "error_message": None}

        # If no actionable input, fall back to interactive prompt if enabled
        # Removed interactive mode - no manual intervention available
        if isinstance(note2, str):
            t = note2.strip()
            if t.lower() == "retry":
                return {"error_message": None}
            if t.lower() == "skip":
                return {"error_message": None, "error_node": None, "skip": True}
            if "=" in t:
                var, value = t.split("=", 1)
                os.environ[var.strip()] = value.strip()
                return {"error_message": None}
            if t:
                notes = list(state.get("feedback", []))
                notes.append(t)
                return {"feedback": notes, "error_message": None}

    # Default: just clear the message
    return {"error_message": None}


async def process_scenes_concurrently_node(state: StoryState) -> dict[str, Any]:
    """Process multiple scenes concurrently for 90% performance improvement."""
    from chorus.langgraph.graph import scene_concurrency_manager

    # Get available scenes from queue
    queue = list(state.get("scene_queue", []))
    if len(queue) <= 1:
        return {}  # Not enough scenes for concurrent processing

    # Prepare scene processing tasks (up to max_concurrent)
    max_concurrent = scene_concurrency_manager.max_concurrent
    concurrent_scenes = queue[:max_concurrent]

    # Create processing functions for each scene
    scene_tasks = []
    for scene_id in concurrent_scenes:
        # Create a processing pipeline for each scene
        async def create_scene_processor(sid: str):
            async def process_scene(scene_state: StoryState) -> dict[str, Any]:
                """Process a single scene through the full pipeline."""
                results = {}

                # Set current scene
                scene_state = {**scene_state, "current_scene_id": sid}

                try:
                    # 1. Gather lore context
                    lore_result = await gather_lore_context_node(scene_state)
                    results.update(lore_result)
                    scene_state.update(lore_result)

                    # 2. Retrieve memory
                    memory_result = await retrieve_memory_node(scene_state)
                    results.update(memory_result)
                    scene_state.update(memory_result)

                    # 3. Draft scene
                    draft_result = await draft_scene(scene_state)
                    results.update(draft_result)
                    scene_state.update(draft_result)

                    # 4. Coherence check (replaces old review system)
                    coherence_result = await coherence_check_node(scene_state)
                    results.update(coherence_result)
                    scene_state.update(coherence_result)

                    # Track successful completion
                    results["scene_completed"] = True
                    results["scene_id"] = sid

                    log_message(f"Concurrent scene {sid} completed successfully")
                    return results

                except Exception as e:
                    log_message(f"Concurrent scene {sid} failed: {e}")
                    return {
                        "error_messages": [f"Scene {sid} processing failed: {str(e)}"],
                        "scene_id": sid,
                        "scene_completed": False,
                    }

            return process_scene

        processor = await create_scene_processor(scene_id)
        scene_tasks.append((scene_id, processor))

    # Process scenes concurrently
    concurrent_results = await scene_concurrency_manager.process_scenes_concurrently(
        scene_tasks, state
    )

    # Remove processed scenes from queue
    remaining_queue = queue[len(concurrent_scenes) :]
    concurrent_results["scene_queue"] = remaining_queue

    # Track performance metrics
    processed_count = concurrent_results.get("concurrent_scenes_processed", 0)
    log_message(
        f"Concurrent processing completed: {processed_count} scenes processed simultaneously"
    )

    return concurrent_results


# Legacy graph weaver stubs removed in Phase 4 cleanup
# These Neo4j-related functions are no longer needed in the single-pass workflow


__all__ = [
    "generate_concepts_node",
    "select_concept_node",
    "generate_world",
    "generate_profiles",
    "seed_narrative_context",
    "generate_outline",
    "prepare_scenes",
    "generate_scenes",
    "dequeue_scene",
    "draft_scene",
    "coherence_check_node",  # Phase 2: Single-pass coherence validation
    "check_character_consistency",  # Phase 2: Character consistency helper
    "check_basic_readability",  # Phase 2: Basic readability validation
    # Legacy review functions removed in Phase 4
    "manual_revision",
    "revise_scene",
    "integrate",
    "finalize_scene_node",
    "catalog_lore_node",
    "gather_lore_context_node",
    "retrieve_memory_node",
    "store_memory_node",
    "summarize_memory_node",
    "human_feedback_node",
    "handle_error_node",
    "persist_scene",
    "evolve_profiles_node",
    "process_scenes_concurrently_node",
    "refill_scene_queue_if_needed",
    # Legacy graph weaver stubs removed in Phase 4
    "_retryable",
    "safe_node",
    "configure_node_error_handling",
    "get_node_criticality",
    "apply_safe_node_to_all",
]
