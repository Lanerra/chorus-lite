# src/chorus/langgraph/integration.py
"""Adapters for invoking LangGraph nodes incrementally."""

from __future__ import annotations

from typing import cast

from chorus.models import CharacterProfile, Concept, Story
from chorus.models.task import SceneBrief

from .state import StoryState


async def generate_scenes_graph(
    concept: Concept,
    profiles: list[CharacterProfile] | None = None,
) -> tuple[Story, list[SceneBrief]]:
    """Generate and persist a story outline using LangGraph nodes."""

    state: StoryState = {"vision": concept}
    from .nodes import (
        generate_outline,
        prepare_scenes,
        seed_narrative_context,
    )

    if profiles:
        state["character_profiles"] = profiles
        update = await seed_narrative_context(state)
        state.update(cast(StoryState, update))

    update = await generate_outline(state)
    state.update(cast(StoryState, update))
    update = await prepare_scenes(state)
    state.update(cast(StoryState, update))

    outline = cast(Story, state["outline"])
    briefs = cast(list[SceneBrief], state.get("scene_briefs", []))
    return outline, briefs


async def generate_profiles_graph(concept: Concept) -> list[CharacterProfile]:
    """Return character profiles using LangGraph nodes."""

    from .nodes import generate_profiles

    state: StoryState = {"vision": concept}
    update = await generate_profiles(state)
    state.update(cast(StoryState, update))
    return cast(list[CharacterProfile], state.get("character_profiles", []))


async def orchestrate_story_graph(idea: str) -> Concept:
    """Return a selected concept after running the planning nodes.

    This helper runs concept generation and invokes ``select_concept_node`` so
    the user can choose a concept interactively via LangGraph's interrupt
    mechanism. It then proceeds with world, profile, and outline generation to
    produce initial scene briefs.
    """

    state: StoryState = {"idea": idea}

    from .nodes import (
        generate_concepts_node,
        generate_outline,
        generate_profiles,
        generate_world,
        prepare_scenes,
        seed_narrative_context,
        select_concept_node,
    )

    update = await generate_concepts_node(state)
    state.update(cast(StoryState, update))
    update = await select_concept_node(state)
    state.update(cast(StoryState, update))
    vision = cast(Concept, state["vision"])

    update = await generate_world(state)
    state.update(cast(StoryState, update))
    update = await generate_profiles(state)
    state.update(cast(StoryState, update))
    update = await seed_narrative_context(state)
    state.update(cast(StoryState, update))
    update = await generate_outline(state)
    state.update(cast(StoryState, update))
    update = await prepare_scenes(state)
    state.update(cast(StoryState, update))

    return vision


__all__ = [
    "generate_scenes_graph",
    "generate_profiles_graph",
    "orchestrate_story_graph",
]
