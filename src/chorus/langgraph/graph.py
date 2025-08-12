# src/chorus/langgraph/graph.py
"""LangGraph orchestration for story generation.

This module defines the production graph used after the LangGraph migration.
The workflow is split into modular subgraphs for story setup and revision
loops.  Each node is intentionally narrow in scope so developers can extend or
replace functionality without affecting unrelated parts of the graph.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import CachePolicy, RetryPolicy

from chorus.core.logs import get_logger

from ..config import config
from .nodes import (
    catalog_lore_node,
    coherence_check_node,  # Phase 4: Single-pass coherence validation
    dequeue_scene,
    draft_scene,
    evolve_profiles_node,
    finalize_scene_node,
    gather_lore_context_node,
    generate_concepts_node,
    generate_outline,
    generate_profiles,
    generate_world,
    handle_error_node,
    human_feedback_node,
    integrate,
    manual_revision,
    prepare_scenes,
    process_scenes_concurrently_node,
    refill_scene_queue_if_needed,
    retrieve_memory_node,
    revise_scene,
    safe_node,
    seed_narrative_context,
    store_memory_node,
    summarize_memory_node,
)
from .state import StoryState

logger = get_logger(__name__)

from .error_recovery import error_recovery_node

ENABLE_ENHANCED_ERROR_HANDLING = True


# Concurrent scene processing infrastructure
class SceneConcurrencyManager:
    """Manages concurrent scene processing with circuit breakers and semaphore controls."""

    def __init__(self):
        self.max_concurrent = config.concurrency.scene_concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.circuit_breaker_threshold = config.concurrency.circuit_breaker_threshold
        self.circuit_breaker_timeout = config.concurrency.circuit_breaker_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_circuit_open = False

    async def process_scenes_concurrently(
        self, scene_tasks: list[tuple[str, callable]], state: StoryState
    ) -> dict[str, any]:
        """Process multiple scenes concurrently with circuit breaker protection."""

        # Check circuit breaker
        if await self._check_circuit_breaker():
            return {"error_messages": ["Circuit breaker open - too many failures"]}

        # Limit concurrent tasks
        limited_tasks = scene_tasks[: self.max_concurrent]

        async def process_single_scene(
            scene_id: str, scene_func: callable
        ) -> tuple[str, dict]:
            """Process a single scene with semaphore control."""
            async with self.semaphore:
                try:
                    logger.info(f"Starting concurrent processing of scene {scene_id}")
                    scene_state = state.get("scene_states", {}).get(scene_id, {})
                    temp_state = {**state, "current_scene_id": scene_id, **scene_state}
                    result = await scene_func(temp_state)
                    await self._record_success()
                    return scene_id, result
                except Exception as e:
                    await self._record_failure()
                    logger.error(f"Scene {scene_id} processing failed: {e}")
                    return scene_id, {"error_messages": [str(e)]}

        # Execute concurrent scene processing
        try:
            logger.info(f"Processing {len(limited_tasks)} scenes concurrently")
            tasks = [
                process_single_scene(scene_id, func) for scene_id, func in limited_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            scene_states = {}
            errors = []

            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                    await self._record_failure()
                else:
                    scene_id, scene_result = result
                    if "error_messages" in scene_result:
                        errors.extend(scene_result["error_messages"])
                    else:
                        scene_states[scene_id] = scene_result

            return {
                "scene_states": scene_states,
                "error_messages": errors if errors else None,
                "concurrent_scenes_processed": len(scene_states),
            }

        except Exception as e:
            await self._record_failure()
            logger.error(f"Concurrent scene processing failed: {e}")
            return {"error_messages": [f"Concurrent processing error: {e}"]}

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be opened."""
        current_time = time.time()

        # Reset circuit breaker after timeout
        if (
            self.is_circuit_open
            and (current_time - self.last_failure_time) > self.circuit_breaker_timeout
        ):
            self.is_circuit_open = False
            self.failure_count = 0
            logger.info("Circuit breaker reset")

        # Open circuit breaker if threshold exceeded
        if self.failure_count >= self.circuit_breaker_threshold:
            self.is_circuit_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

        return self.is_circuit_open

    async def _record_success(self):
        """Record successful operation."""
        self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery

    async def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()


# Global concurrency manager instance
scene_concurrency_manager = SceneConcurrencyManager()

# Retry policy configuration based on environment variables
RETRY_ATTEMPTS = config.retry.retry_attempts
RETRY_BACKOFF = config.retry.retry_backoff
RETRY_MAX_INTERVAL = config.retry.retry_max_interval

# Cache TTL configuration
CACHE_TTL_WORLD_GEN = config.cache.cache_ttl_world_gen  # 1 hour for world generation
CACHE_TTL_EXPENSIVE = (
    config.cache.cache_ttl_expensive
)  # 30 mins for other expensive ops

# Default retry policy for most operations
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_attempts=RETRY_ATTEMPTS,
    initial_interval=RETRY_BACKOFF,
    backoff_factor=2.0,
    max_interval=RETRY_MAX_INTERVAL,
    jitter=True,
)

# Cache policy for expensive world generation operations
WORLD_GEN_CACHE_POLICY: CachePolicy = CachePolicy(ttl=CACHE_TTL_WORLD_GEN)

# Cache policy for other expensive operations (character generation, scene writing, etc.)
EXPENSIVE_OP_CACHE_POLICY: CachePolicy = CachePolicy(ttl=CACHE_TTL_EXPENSIVE)


def _route_error(target: str) -> Callable[[StoryState], Awaitable[str]]:
    """Return a function routing to ``target`` unless an error occurred."""

    async def _fn(state: StoryState) -> str:
        if state.get("error_messages"):
            return "error_recovery"
        return target

    _fn.__name__ = f"route_error_to_{target}"
    return _fn


async def _route_after_error(state: StoryState) -> str:
    """Return to the node that most recently failed."""
    if state.get("skip"):
        return END
    nodes = state.get("error_nodes") or []
    node = nodes[-1] if nodes else None
    return str(node) if node else END


async def _route_after_enhanced_error_recovery(state: StoryState) -> str:
    """Route after enhanced error recovery based on recovery strategy."""

    # Check recovery strategy result
    recovery_strategy = state.get("recovery_strategy")

    if recovery_strategy == "retry":
        # Return to the failed node for retry
        nodes = state.get("error_nodes") or []
        node = nodes[-1] if nodes else None
        return str(node) if node else END

    elif recovery_strategy == "skip":
        # Skip the failed node and continue to next step
        return _determine_next_node_after_skip(state)

    elif recovery_strategy == "rollback":
        # Return to a previous stable state
        rollback_target = state.get("rollback_target", "dequeue_scene")
        return rollback_target

    elif recovery_strategy == "circuit_break":
        # Circuit breaker is open, pause processing
        return END

    elif recovery_strategy == "manual":
        # Route to manual error handling for human intervention
        return "handle_error"

    elif recovery_strategy == "escalate":
        # Escalate to critical error handling
        return "handle_error"

    else:
        # Default: end processing if no strategy determined
        return END


async def route_after_coherence(state: StoryState) -> str:
    """Simple routing after coherence check - replaces complex review routing."""

    if state.get("error_messages"):
        return "handle_error"

    # Check coherence result
    if state.get("coherence_passed"):
        return "finalize_scene"

    # Check retry limit for failed coherence
    limit = config.retry.retry_attempts
    if state.get("revision_count", 0) < limit:
        return "revise_scene"

    return "manual_revision"


def _determine_next_node_after_skip(state: StoryState) -> str:
    """Determine the next node to proceed to when skipping a failed node."""

    # Get the failed node to determine appropriate next step
    nodes = state.get("error_nodes") or []
    failed_node = nodes[-1] if nodes else None

    # Define skip routing logic based on failed node
    skip_routing = {
        # Optional/enhancement nodes can be skipped gracefully
        "catalog_lore": "summarize_memory",
        "evolve_profiles": "summarize_memory",
        # Context nodes - proceed to next step in pipeline
        "lore_context": "retrieve_memory",
        "retrieve_memory": "draft_scene",
        # Phase 4: Update skip routing for coherence check
        "coherence_check": "finalize_scene",
        # Memory nodes - proceed to next step
        "store_memory": "summarize_memory",
        "summarize_memory": "dequeue_scene",
    }

    next_node = skip_routing.get(failed_node, "dequeue_scene")
    logger.info(f"Skipping failed node '{failed_node}', proceeding to '{next_node}'")
    return next_node


def _add_story_setup(builder: StateGraph) -> None:
    """Add the story setup subgraph to ``builder``."""
    logger.info("Adding story setup subgraph")
    # Story setup nodes with retry policies and caching for expensive operations
    builder.add_node(
        "generate_concepts",
        safe_node(generate_concepts_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
        cache_policy=EXPENSIVE_OP_CACHE_POLICY,
    )
    # Remove select_concept node since we're directly using the first concept
    # Add human_feedback node for user interaction
    builder.add_node(
        "human_feedback", safe_node(human_feedback_node, ENABLE_ENHANCED_ERROR_HANDLING)
    )  # No retry needed for user interaction
    builder.add_node(
        "generate_world",
        safe_node(generate_world, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
        cache_policy=WORLD_GEN_CACHE_POLICY,  # Longer cache for world generation
    )
    # Removed weave_world_graph reference as it doesn't exist in nodes.py
    # This was a leftover from previous implementation and caused import error
    builder.add_node(
        "generate_profiles",
        safe_node(generate_profiles, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
        cache_policy=EXPENSIVE_OP_CACHE_POLICY,
    )
    builder.add_node(
        "seed_narrative_context",
        safe_node(seed_narrative_context, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
        cache_policy=EXPENSIVE_OP_CACHE_POLICY,
    )
    builder.add_node(
        "generate_outline",
        safe_node(generate_outline, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
        cache_policy=EXPENSIVE_OP_CACHE_POLICY,
    )
    builder.add_node(
        "prepare_scenes",
        safe_node(prepare_scenes, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    builder.add_node(
        "dequeue_scene", safe_node(dequeue_scene, ENABLE_ENHANCED_ERROR_HANDLING)
    )  # Fast operation, no retry needed
    builder.add_node(
        "refill_scene_queue",
        safe_node(refill_scene_queue_if_needed, ENABLE_ENHANCED_ERROR_HANDLING),
    )  # Top up queue to act as a perpetual worker
    builder.add_node(
        "handle_error", safe_node(handle_error_node, ENABLE_ENHANCED_ERROR_HANDLING)
    )  # Error handler, no retry

    # Add enhanced error recovery node if available
    if ENABLE_ENHANCED_ERROR_HANDLING:
        builder.add_node(
            "error_recovery",
            safe_node(error_recovery_node, ENABLE_ENHANCED_ERROR_HANDLING),
            retry_policy=DEFAULT_RETRY_POLICY,  # Error recovery itself can be retried
        )
        logger.info("Added enhanced error recovery node to graph")

    builder.add_edge(START, "generate_concepts")
    # Directly route from generate_concepts to human_feedback since we're using the first concept
    # Directly route from generate_concepts to human_feedback since we're using the first concept
    builder.add_conditional_edges("generate_concepts", _route_error("human_feedback"))

    async def route_after_select(state: StoryState) -> str:
        """Sequentialize setup after concept selection."""
        if state.get("error_messages"):
            return "handle_error"
        if state.get("concept_rejected"):
            return "human_feedback"
        return "generate_world"

    # Removed select_concept edge since we're directly using the first concept
    builder.add_conditional_edges("human_feedback", _route_error("generate_concepts"))

    # Sequential setup: world -> profiles -> seed -> outline -> prepare_scenes -> dequeue
    builder.add_conditional_edges("generate_world", _route_error("generate_profiles"))
    builder.add_conditional_edges(
        "generate_profiles", _route_error("seed_narrative_context")
    )
    builder.add_conditional_edges(
        "seed_narrative_context", _route_error("generate_outline")
    )
    builder.add_conditional_edges("generate_outline", _route_error("prepare_scenes"))
    builder.add_conditional_edges("prepare_scenes", _route_error("dequeue_scene"))

    async def route_after_dequeue(state: StoryState) -> str:
        """Return ``lore_context`` if a scene is available else try refilling."""

        if state.get("error_messages"):
            return "handle_error"
        # If a scene is available, proceed; otherwise attempt to refill the queue.
        return "lore_context" if state.get("current_scene_id") else "refill_scene_queue"

    builder.add_conditional_edges("dequeue_scene", route_after_dequeue)

    async def route_after_refill(state: StoryState) -> str:
        """After attempting to refill the queue, either proceed or end."""
        if state.get("error_messages"):
            return "handle_error"
        # If queue still empty after refill, end; otherwise try dequeue again
        queue = state.get("scene_queue") or []
        return "dequeue_scene" if queue else END

    # Add concurrent scene processing node
    builder.add_node(
        "process_scenes_concurrently",
        safe_node(process_scenes_concurrently_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )

    async def route_to_concurrent_processing(state: StoryState) -> str:
        """Route to concurrent processing if multiple scenes available."""
        if state.get("error_messages"):
            return "handle_error"

        queue = state.get("scene_queue", [])
        # Use concurrent processing if we have multiple scenes and it's enabled
        if len(queue) > 1 and config.concurrency.scene_concurrency > 1:
            return "process_scenes_concurrently"
        return "lore_context"  # Single scene processing

    # Update routing to use concurrent processing
    builder.add_conditional_edges("dequeue_scene", route_to_concurrent_processing)
    builder.add_conditional_edges(
        "process_scenes_concurrently", _route_error("dequeue_scene")
    )

    builder.add_conditional_edges("refill_scene_queue", route_after_refill)
    builder.add_conditional_edges("handle_error", _route_after_error)

    # Add enhanced error recovery routing if available
    if ENABLE_ENHANCED_ERROR_HANDLING:
        builder.add_conditional_edges(
            "error_recovery", _route_after_enhanced_error_recovery
        )
        logger.info("Added enhanced error recovery routing to graph")


def _add_revision_loop(builder: StateGraph) -> None:
    """Add the scene drafting and revision subgraph to ``builder``."""
    logger.info("Adding revision loop subgraph")

    # Scene revision nodes with retry policies and caching where appropriate
    builder.add_node(
        "lore_context",
        safe_node(gather_lore_context_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    builder.add_node(
        "retrieve_memory",
        safe_node(retrieve_memory_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    # Legacy weave_scene_graph removed in Phase 4 cleanup
    builder.add_node(
        "draft_scene",
        safe_node(draft_scene, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
        cache_policy=EXPENSIVE_OP_CACHE_POLICY,  # Scene writing is expensive
    )
    # Phase 3: Add single coherence check node replacing multi-stage reviews
    builder.add_node(
        "coherence_check",
        safe_node(coherence_check_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    # Legacy review nodes removed in Phase 4 cleanup
    builder.add_node(
        "revise_scene",
        safe_node(revise_scene, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    builder.add_node(
        "manual_revision", safe_node(manual_revision, ENABLE_ENHANCED_ERROR_HANDLING)
    )  # User interaction, no retry
    builder.add_node(
        "integrate",
        safe_node(integrate, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    # Legacy select_revision_node removed in Phase 4 cleanup
    builder.add_node(
        "finalize_scene",
        safe_node(finalize_scene_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    builder.add_node(
        "store_memory",
        safe_node(store_memory_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    builder.add_node(
        "summarize_memory",
        safe_node(summarize_memory_node, ENABLE_ENHANCED_ERROR_HANDLING),
    )  # Fast operation
    # Legacy canonize_scene_graph_node and detect_character_clusters_node removed in Phase 4 cleanup
    builder.add_node(
        "catalog_lore",
        safe_node(catalog_lore_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )
    # Legacy weave_lore_graph removed in Phase 4 cleanup
    builder.add_node(
        "evolve_profiles",
        safe_node(evolve_profiles_node, ENABLE_ENHANCED_ERROR_HANDLING),
        retry_policy=DEFAULT_RETRY_POLICY,
    )

    # Phase 3: Linear workflow: lore_context -> retrieve_memory -> draft_scene -> coherence_check -> finalize/retry
    builder.add_conditional_edges("lore_context", _route_error("retrieve_memory"))
    builder.add_conditional_edges("retrieve_memory", _route_error("draft_scene"))
    builder.add_conditional_edges("draft_scene", _route_error("coherence_check"))
    builder.add_conditional_edges("coherence_check", route_after_coherence)
    builder.add_conditional_edges("revise_scene", _route_error("draft_scene"))

    # Legacy review workflow removed in Phase 4 cleanup

    async def route_after_manual(state: StoryState) -> str:
        """Route after manual revision input."""
        if state.get("error_messages"):
            return "handle_error"
        # Phase 3: Route to coherence check instead of integrate for simplified workflow
        return "coherence_check" if state.get("needs_revision") else "finalize_scene"

    builder.add_conditional_edges("manual_revision", route_after_manual)

    async def route_after_integrate(state: StoryState) -> str:
        """Restart planning when the final draft is rejected."""
        if state.get("error_messages"):
            return "handle_error"
        if state.get("draft_rejected"):
            return "human_feedback"
        return "finalize_scene"  # Simplified without select_revision

    builder.add_conditional_edges("integrate", route_after_integrate)

    builder.add_conditional_edges("finalize_scene", _route_error("store_memory"))
    builder.add_conditional_edges("store_memory", _route_error("catalog_lore"))
    builder.add_conditional_edges("finalize_scene", _route_error("evolve_profiles"))
    builder.add_edge(["catalog_lore", "evolve_profiles"], "summarize_memory")
    # After summarizing memory, loop back to dequeue next scene; refill logic will keep it alive
    builder.add_conditional_edges("summarize_memory", _route_error("dequeue_scene"))


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Return a compiled LangGraph state graph."""
    logger.info("Building LangGraph state graph")
    t0 = time.perf_counter()

    builder = StateGraph(StoryState)

    # Pre-add logging: environment gates that can prune nodes
    logger.info(
        "Graph gates | retry_attempts=%s | cache_ttl_world=%s | cache_ttl_expensive=%s",
        RETRY_ATTEMPTS,
        CACHE_TTL_WORLD_GEN,
        CACHE_TTL_EXPENSIVE,
    )

    _add_story_setup(builder)
    _add_revision_loop(builder)

    compiled = builder.compile(checkpointer=checkpointer)
    duration = time.perf_counter() - t0

    logger.info("LangGraph compiled | duration=%.2fs", duration)
    return compiled


__all__ = [
    "build_graph",
    "_add_story_setup",
    "_add_revision_loop",
    "_route_after_enhanced_error_recovery",
    "_determine_next_node_after_skip",
    "scene_concurrency_manager",
    "route_after_coherence",  # Phase 3: Export coherence routing function
]
