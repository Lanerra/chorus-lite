# src/chorus/agents/graph_node_utils.py
"""Utility functions for calling LangGraph nodes from agent classes."""

from __future__ import annotations

from typing import Any, cast

from chorus.canon.postgres import get_pg, get_scene
from chorus.core.logs import get_event_logger, EventType, LogLevel, Priority
from chorus.models import CharacterProfile, Concept, WorldAnvil
from chorus.models.task import SceneBrief

# NOTE: Avoid importing chorus.langgraph at module import time to prevent circular imports.
# Import StoryState type only for typing (no runtime import), and construct dicts at runtime.
try:  # typing-only import to avoid runtime circular import
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        pass  # pragma: no cover - typing only
except Exception:
    pass


def _parse_text_dummy(
    text: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Dummy implementation that returns empty lists."""
    return ([], [], [], [], [])


async def call_graph_node_draft_scene(
    scene_id: str,
    model: str | None = None,
    extra_context: str | None = None,
) -> dict[str, Any]:
    """Call the draft_scene LangGraph node with appropriate state.

    This function must provide a SceneBrief entry matching current_scene_id
    because nodes._current_brief() searches state['scene_briefs'] for a brief
    whose id equals current_scene_id. Without it, the node raises
    ValueError("scene brief missing id") and the worker loops.
    """
    event_logger = get_event_logger()
    
    event_logger.log(
        LogLevel.INFO,
        f"Starting draft_scene node call for scene: {scene_id}",
        event_type=EventType.AGENT_OPERATION,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_draft_scene",
            "scene_id": scene_id,
            "model": model,
            "has_extra_context": extra_context is not None
        }
    )
    
    # Localize imports to avoid circular imports at module import time
    from chorus.config import config  # localized import
    from chorus.langgraph.nodes import draft_scene  # localized import

    if model:
        config.agents.scene_generator = model
        event_logger.log(
            LogLevel.DEBUG,
            f"Set scene generator model to: {model}",
            event_type=EventType.AGENT_OPERATION,
            priority=Priority.NORMAL,
            metadata={"operation": "call_graph_node_draft_scene", "model": model}
        )

    # Build a minimal SceneBrief for the given scene_id to satisfy _current_brief.
    # Prefer loading from the DB; if missing, synthesize placeholders.
    briefs: list[SceneBrief] = []
    try:
        event_logger.log(
            LogLevel.DEBUG,
            f"Loading scene data from database for scene: {scene_id}",
            event_type=EventType.DATABASE_OPERATION,
            priority=Priority.NORMAL,
            metadata={"operation": "call_graph_node_draft_scene", "scene_id": scene_id}
        )
        async with get_pg() as conn:
            scene = await get_scene(conn, scene_id)
    except Exception as e:
        event_logger.log(
            LogLevel.WARNING,
            f"Failed to load scene {scene_id} from database: {e}",
            event_type=EventType.DATABASE_OPERATION,
            priority=Priority.NORMAL,
            metadata={
                "operation": "call_graph_node_draft_scene",
                "scene_id": scene_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        scene = None  # fall through to placeholder

    if scene is not None:
        # Convert character UUIDs to strings; provide at least one placeholder
        characters: list[str] = [str(c) for c in scene.characters]
        if not characters:
            # Attempt to parse characters from description as a fallback
            try:
                parsed_chars, _, _, _, _ = _parse_text_dummy(scene.description or "")
            except Exception:
                parsed_chars = []
            characters = sorted(parsed_chars) if parsed_chars else ["Unknown"]
        intentions: list[str] = [scene.description] if scene.description else ["TBD"]
        briefs.append(
            SceneBrief(
                id=scene.id,
                title=scene.title or "Untitled",
                description=scene.description or "No description.",
                characters=characters,
                intentions=intentions,
            )
        )
        event_logger.log(
            LogLevel.DEBUG,
            f"Created scene brief from database scene: '{scene.title}' with {len(characters)} characters",
            event_type=EventType.AGENT_OPERATION,
            priority=Priority.NORMAL,
            metadata={
                "operation": "call_graph_node_draft_scene",
                "scene_id": scene_id,
                "scene_title": scene.title,
                "characters_count": len(characters)
            }
        )
    else:
        # Synthesize a placeholder brief to unblock the node
        briefs.append(
            SceneBrief(
                id=scene_id,  # type: ignore[arg-type]
                title="Untitled",
                description="No description.",
                characters=["Unknown"],
                intentions=["TBD"],
            )
        )
        event_logger.log(
            LogLevel.WARNING,
            f"Created placeholder scene brief for missing scene: {scene_id}",
            event_type=EventType.AGENT_OPERATION,
            priority=Priority.NORMAL,
            metadata={"operation": "call_graph_node_draft_scene", "scene_id": scene_id}
        )

    # Create state for the node (use plain dict to avoid runtime type import)
    state = {
        "current_scene_id": scene_id,
        "scene_briefs": briefs,
        "memory_summary": None,
        "lore_context": None,
        "memory_context": extra_context or None,
        "character_clusters": [],
        "scene_states": {scene_id: {}},
    }

    event_logger.log(
        LogLevel.DEBUG,
        f"Calling draft_scene LangGraph node for scene: {scene_id}",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_draft_scene",
            "scene_id": scene_id,
            "node": "draft_scene",
            "briefs_count": len(briefs)
        }
    )

    result = await draft_scene(state)  # type: ignore[arg-type]
    
    event_logger.log(
        LogLevel.INFO,
        f"Successfully completed draft_scene node call for scene: {scene_id}",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_draft_scene",
            "scene_id": scene_id,
            "node": "draft_scene",
            "result_keys": list(result.keys()) if isinstance(result, dict) else "non-dict"
        }
    )
    
    return result


async def call_graph_node_integrate(
    scene_id: str, draft: str, feedback: list[str], model: str | None = None
) -> dict[str, Any]:
    """Call the integrate LangGraph node with appropriate state."""
    event_logger = get_event_logger()
    
    event_logger.log(
        LogLevel.INFO,
        f"Starting integrate node call for scene: {scene_id}",
        event_type=EventType.AGENT_OPERATION,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_integrate",
            "scene_id": scene_id,
            "model": model,
            "feedback_count": len(feedback),
            "draft_length": len(draft)
        }
    )
    
    from chorus.config import config  # localized import
    from chorus.langgraph.nodes import integrate  # localized import

    if model:
        config.agents.integration_manager = model

    async with get_pg() as conn:
        scene = await get_scene(conn, scene_id)

    briefs: list[SceneBrief] = []
    if scene is not None:
        # Convert character UUIDs to strings and fall back to names parsed
        # from the draft if none are provided. Provide a placeholder if no
        # characters are found to satisfy validation requirements.
        characters: list[str] = [str(c) for c in scene.characters]
        if not characters:
            parsed_chars, _, _, _, _ = _parse_text_dummy(draft)
            characters = sorted(parsed_chars) if parsed_chars else ["Unknown"]

        # Use the scene description as a default intention when none are
        # available, ensuring the SceneBrief passes validation.
        intentions: list[str] = [scene.description] if scene.description else ["TBD"]

        briefs.append(
            SceneBrief(
                id=scene.id,
                title=scene.title or "Untitled",
                description=scene.description or "No description.",
                characters=characters,
                intentions=intentions,
            )
        )

    state = {
        "current_scene_id": scene_id,
        "draft": draft,
        "feedback": feedback,
        "revision_notes": [],
        "scene_briefs": briefs,
        "scene_states": {scene_id: {}},
    }

    event_logger.log(
        LogLevel.DEBUG,
        f"Calling integrate LangGraph node for scene: {scene_id}",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_integrate",
            "scene_id": scene_id,
            "node": "integrate",
            "briefs_count": len(briefs)
        }
    )

    result = await integrate(state)  # type: ignore[arg-type]
    
    event_logger.log(
        LogLevel.INFO,
        f"Successfully completed integrate node call for scene: {scene_id}",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_integrate",
            "scene_id": scene_id,
            "node": "integrate"
        }
    )
    
    return result


async def call_graph_node_revise_scene(
    scene_id: str, draft: str, feedback: list[str], model: str | None = None
) -> dict[str, Any]:
    """Call the revise_scene LangGraph node with appropriate state."""
    event_logger = get_event_logger()
    
    event_logger.log(
        LogLevel.INFO,
        f"Starting revise_scene node call for scene: {scene_id}",
        event_type=EventType.AGENT_OPERATION,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_revise_scene",
            "scene_id": scene_id,
            "model": model,
            "feedback_count": len(feedback),
            "draft_length": len(draft)
        }
    )
    
    from chorus.config import config  # localized import
    from chorus.langgraph.nodes import revise_scene  # localized import

    if model:
        config.agents.integration_manager = model

    async with get_pg() as conn:
        scene = await get_scene(conn, scene_id)

    briefs: list[SceneBrief] = []
    if scene is not None:
        characters: list[str] = [str(c) for c in scene.characters]
        if not characters:
            parsed_chars, _, _, _, _ = _parse_text_dummy(draft)
            characters = sorted(parsed_chars) if parsed_chars else ["Unknown"]

        intentions: list[str] = [scene.description] if scene.description else ["TBD"]

        briefs.append(
            SceneBrief(
                id=scene.id,
                title=scene.title or "Untitled",
                description=scene.description or "No description.",
                characters=characters,
                intentions=intentions,
            )
        )

    state = {
        "current_scene_id": scene_id,
        "draft": draft,
        "feedback": feedback,
        "revision_notes": [],
        "revision_history": [],
        "scene_briefs": briefs,
        "scene_states": {scene_id: {}},
    }

    event_logger.log(
        LogLevel.DEBUG,
        f"Calling revise_scene LangGraph node for scene: {scene_id}",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_revise_scene",
            "scene_id": scene_id,
            "node": "revise_scene",
            "briefs_count": len(briefs)
        }
    )

    result = await revise_scene(state)  # type: ignore[arg-type]
    
    event_logger.log(
        LogLevel.INFO,
        f"Successfully completed revise_scene node call for scene: {scene_id}",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_revise_scene",
            "scene_id": scene_id,
            "node": "revise_scene"
        }
    )
    
    return result


async def call_graph_node_generate_world(
    concept: Concept, model: str | None = None
) -> list[WorldAnvil]:
    """Call the generate_world LangGraph node."""
    event_logger = get_event_logger()
    
    event_logger.log(
        LogLevel.INFO,
        f"Starting generate_world node call for concept: '{concept.title}'",
        event_type=EventType.AGENT_OPERATION,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_generate_world",
            "concept_title": concept.title,
            "model": model
        }
    )
    
    from chorus.config import config  # localized import
    from chorus.langgraph.nodes import generate_world  # localized import

    if model:
        config.agents.story_architect = model

    state = {"vision": concept}
    
    event_logger.log(
        LogLevel.DEBUG,
        f"Calling generate_world LangGraph node for concept: '{concept.title}'",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_generate_world",
            "concept_title": concept.title,
            "node": "generate_world"
        }
    )
    
    result = await generate_world(state)  # type: ignore[arg-type]
    world_info = cast(list[WorldAnvil], result.get("world_info", []))
    
    event_logger.log(
        LogLevel.INFO,
        f"Successfully completed generate_world node call, generated {len(world_info)} world elements",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_generate_world",
            "concept_title": concept.title,
            "node": "generate_world",
            "world_elements_count": len(world_info)
        }
    )
    
    return world_info


async def call_graph_node_generate_profiles(
    concept: Concept, model: str | None = None
) -> list[CharacterProfile]:
    """Call the generate_profiles LangGraph node."""
    event_logger = get_event_logger()
    
    event_logger.log(
        LogLevel.INFO,
        f"Starting generate_profiles node call for concept: '{concept.title}'",
        event_type=EventType.AGENT_OPERATION,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_generate_profiles",
            "concept_title": concept.title,
            "model": model
        }
    )
    
    import os  # localized import

    from chorus.langgraph.nodes import generate_profiles  # localized import

    if model:
        os.environ["STORY_ARCHITECT_MODEL"] = model

    state = {"vision": concept}
    
    event_logger.log(
        LogLevel.DEBUG,
        f"Calling generate_profiles LangGraph node for concept: '{concept.title}'",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_generate_profiles",
            "concept_title": concept.title,
            "node": "generate_profiles"
        }
    )
    
    result = await generate_profiles(state)  # type: ignore[arg-type]
    profiles = cast(list[CharacterProfile], result.get("character_profiles", []))
    
    event_logger.log(
        LogLevel.INFO,
        f"Successfully completed generate_profiles node call, generated {len(profiles)} character profiles",
        event_type=EventType.LANGGRAPH_NODE,
        priority=Priority.HIGH,
        metadata={
            "operation": "call_graph_node_generate_profiles",
            "concept_title": concept.title,
            "node": "generate_profiles",
            "profiles_count": len(profiles)
        }
    )
    
    return profiles


__all__ = [
    "call_graph_node_draft_scene",
    "call_graph_node_integrate",
    "call_graph_node_revise_scene",
    "call_graph_node_generate_world",
    "call_graph_node_generate_profiles",
]
