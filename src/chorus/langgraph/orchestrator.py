# src/chorus/langgraph/orchestrator.py
"""Public APIs for running the LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RunnableConfig

# Eagerly load .env so CLI and programmatic entry points always see variables
try:
    from chorus.core.env import load_env  # type: ignore

    load_env()
except Exception:
    # Tests may patch env or skip dotenv; fail open
    pass

from chorus.core.logs import get_logger

from .checkpointer import get_checkpointer
from .graph import build_graph
from .session_utils import (
    create_scene_thread_id,
    create_session_bound_thread_id,
    extract_session_id_from_thread_id,
)
from .state import SessionManager, StoryState

logger = get_logger(__name__)

# Global session manager instance
session_manager = SessionManager()


def _get_checkpointer_context() -> AsyncContextManager:
    _cm = get_checkpointer()
    if hasattr(_cm, "__aenter__"):
        return _cm

    @asynccontextmanager
    async def _async_wrap() -> AsyncIterator[Any]:
        with _cm as s:  # type: ignore[attr-defined, misc]
            yield s

    return _async_wrap()


def _setup_recursion_limit(config: RunnableConfig, recursion_limit: int | None) -> None:
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit


async def create_story(
    idea: str, *, session_id: str | None = None, recursion_limit: int | None = None
) -> tuple[str, StoryState]:
    """Start a new story and return the thread id and final state.

    Args:
        idea: The story idea to develop
        session_id: Optional session identifier for state isolation. If not provided,
                   a new session will be created.
        recursion_limit: Optional recursion limit for the graph execution

    Returns:
        Tuple of (thread_id, final_state)
    """

    # Defensive: ensure env is loaded even if import-time hook was skipped
    try:
        from chorus.core.env import load_env as _load_env  # type: ignore

        _load_env()
    except Exception:
        pass

    # Create session-bound thread ID for complete isolation
    thread_id = create_session_bound_thread_id(session_id)

    async with _get_checkpointer_context() as saver:
        logger.info(
            "Orchestrator.create_story | new session_id=%s thread_id=%s",
            session_id,
            thread_id,
        )

        # Register thread with session manager
        if session_id is not None:
            session_manager.register_thread(session_id, thread_id)
            session_manager.set_session_status(session_id, "creating_story")

        graph = build_graph(checkpointer=saver)
        try:
            # Best-effort introspection of compiled graph.
            try:
                # Newer/older langgraph expose .graph or .compiled_graph in different ways.
                g_attr = getattr(graph, "graph", None)
                if hasattr(graph, "nodes") and hasattr(graph, "edges"):
                    logger.info(
                        "Compiled graph meta (direct) | nodes=%s edges=%s",
                        len(graph.nodes),
                        len(graph.edges),
                    )
                elif g_attr is not None:
                    nodes = getattr(g_attr, "nodes", None) or getattr(
                        g_attr, "_nodes", None
                    )
                    edges = getattr(g_attr, "edges", None) or getattr(
                        g_attr, "_edges", None
                    )
                    logger.info(
                        "Compiled graph meta (g_attr) | nodes=%s edges=%s",
                        len(nodes) if nodes is not None else "?",
                        len(edges) if edges is not None else "?",
                    )
                else:
                    logger.info("Compiled graph meta | no graph attribute found")
            except Exception as e:
                logger.warning("Failed to introspect compiled graph: %s", e)
        except Exception:
            pass
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        _setup_recursion_limit(config, recursion_limit)

        try:
            # Ensure initial state includes both thread_id and session_id for isolation
            initial: StoryState = {
                "idea": idea,
                "thread_id": thread_id,
                "session_id": session_id,
            }
            logger.info(
                "Orchestrator.create_story | invoking graph with initial keys=%s",
                list(initial.keys()),
            )

            # Update session state with initial story data
            if session_id is not None:
                session_manager.update_session_state(
                    session_id,
                    {"idea": idea, "thread_id": thread_id, "status": "in_progress"},
                )

            state = await graph.ainvoke(initial, config)

            # Update session state with final results
            if session_id is not None:
                session_manager.update_session_state(
                    session_id, cast(dict[str, Any], state)
                )
                session_manager.set_session_status(session_id, "completed")

        except GraphRecursionError:
            snapshot = await graph.aget_state(config)
            # StateSnapshot is immutable; copy its underlying values to a plain dict
            if hasattr(snapshot, "values"):
                base = cast(dict[str, Any], snapshot.values)  # type: ignore[assignment]
            else:
                # Fallback: try to coerce to dict if it's already mapping-like
                try:
                    base = dict(snapshot)  # type: ignore[arg-type]
                except Exception:
                    base = {}  # last resort
            mutable: dict[str, Any] = {**base}
            mutable["error_messages"] = ["graph recursion limit reached"]
            mutable["error_nodes"] = ["recursion_limit"]
            mutable["session_id"] = session_id

            # Update session with error state
            if session_id is not None:
                session_manager.update_session_state(session_id, mutable)
                session_manager.set_session_status(session_id, "error")

            return thread_id, cast(StoryState, mutable)
        except Exception as e:
            # Handle any other errors
            logger.error(
                "Orchestrator.create_story | error in session %s: %s", session_id, e
            )
            if session_id is not None:
                session_manager.set_session_status(session_id, "error")
            raise
        finally:
            # Always unregister thread on completion
            if session_id is not None:
                session_manager.unregister_thread(session_id, thread_id)

    return thread_id, cast(StoryState, state)


async def resume_story(
    thread_id: str,
    resume_value: Any | None = None,
    *,
    session_id: str | None = None,
    recursion_limit: int | None = None,
) -> StoryState:
    """Resume an existing story identified by ``thread_id``.

    Args:
        thread_id: The thread ID to resume
        resume_value: Optional resume value for the graph
        session_id: Optional session ID for state isolation. If not provided,
                   will attempt to extract from thread_id if it follows the
                   session_bound format.
        recursion_limit: Optional recursion limit for the graph execution

    Returns:
        The current story state
    """

    # Defensive: ensure env is loaded even if import-time hook was skipped
    try:
        from chorus.core.env import load_env as _load_env  # type: ignore

        _load_env()
    except Exception:
        pass

    async with _get_checkpointer_context() as saver:
        logger.info(
            "Orchestrator.resume_story | session_id=%s thread_id=%s",
            session_id,
            thread_id,
        )

        # Register thread with session manager if session_id is available
        if session_id:
            session_manager.register_thread(session_id, thread_id)
            session_manager.set_session_status(session_id, "resuming")

        graph = build_graph(checkpointer=saver)
        try:
            # Introspect compiled graph best-effort.
            try:
                g_attr = getattr(graph, "graph", None)
                if hasattr(graph, "nodes") and hasattr(graph, "edges"):
                    logger.info(
                        "Compiled graph meta (direct) | nodes=%s edges=%s",
                        len(graph.nodes),
                        len(graph.edges),
                    )
                elif g_attr is not None:
                    nodes = getattr(g_attr, "nodes", None) or getattr(
                        g_attr, "_nodes", None
                    )
                    edges = getattr(g_attr, "edges", None) or getattr(
                        g_attr, "_edges", None
                    )
                    logger.info(
                        "Compiled graph meta (g_attr) | nodes=%s edges=%s",
                        len(nodes) if nodes is not None else "?",
                        len(edges) if edges is not None else "?",
                    )
                else:
                    logger.info("Compiled graph meta | no graph attribute found")
            except Exception as e:
                logger.warning("Failed to introspect compiled graph: %s", e)
        except Exception:
            pass
        payload: Command = (
            Command(resume=resume_value) if resume_value is not None else Command()
        )
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        _setup_recursion_limit(config, recursion_limit)

        try:
            logger.info(
                "Orchestrator.resume_story | invoking graph (resume=%s)",
                resume_value is not None,
            )
            state = await graph.ainvoke(payload, config)

            # Update session state if session_id is available
            if session_id:
                session_manager.update_session_state(
                    session_id, cast(dict[str, Any], state)
                )
                session_manager.set_session_status(session_id, "completed")

        except GraphRecursionError:
            snapshot = await graph.aget_state(config)
            # StateSnapshot is immutable; copy its underlying values to a plain dict
            if hasattr(snapshot, "values"):
                base = cast(dict[str, Any], snapshot.values)  # type: ignore[assignment]
            else:
                try:
                    base = dict(snapshot)  # type: ignore[arg-type]
                except Exception:
                    base = {}
            mutable: dict[str, Any] = {**base}
            mutable["error_messages"] = ["graph recursion limit reached"]
            mutable["error_nodes"] = ["recursion_limit"]

            if session_id:
                mutable["session_id"] = session_id
                session_manager.update_session_state(session_id, mutable)
                session_manager.set_session_status(session_id, "error")

            return cast(StoryState, mutable)
        except Exception as e:
            logger.error(
                "Orchestrator.resume_story | error in session %s: %s", session_id, e
            )
            if session_id:
                session_manager.set_session_status(session_id, "error")
            raise
        finally:
            # Always unregister thread on completion
            if session_id:
                session_manager.unregister_thread(session_id, thread_id)

    return cast(StoryState, state)


async def get_scene_status(
    thread_id: str, *, session_id: str | None = None
) -> StoryState:
    """Return the current state for ``thread_id``.

    Args:
        thread_id: The thread ID to get status for
        session_id: Optional session ID for state isolation. If not provided,
                   will attempt to extract from thread_id if it follows the
                   session_bound format.

    Returns:
        The current story state
    """

    # Extract session_id from thread_id if not provided and thread follows session format
    if session_id is None:
        session_id = extract_session_id_from_thread_id(thread_id)

    async with _get_checkpointer_context() as saver:
        logger.info(
            "Orchestrator.get_scene_status | session_id=%s thread_id=%s",
            session_id,
            thread_id,
        )
        graph = build_graph(checkpointer=saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        # Prefer raw values dict for external consumers
        if hasattr(snapshot, "values"):
            state = cast(StoryState, snapshot.values)
        else:
            state = cast(StoryState, snapshot)

        # Ensure session_id is included in the state if available
        if session_id and isinstance(state, dict):
            state["session_id"] = session_id

        return state


async def run_scene_threads(
    base_state: StoryState,
    scene_ids: Iterable[str],
    *,
    max_concurrency: int | None = None,
) -> list[StoryState]:
    """Process ``scene_ids`` concurrently using separate graph threads.

    The ``max_concurrency`` argument or ``SCENE_CONCURRENCY`` environment
    variable controls how many threads run at once. Now supports session-bound
    thread creation for proper isolation.
    """

    limit = max_concurrency or int(os.getenv("SCENE_CONCURRENCY", "0")) or None
    session_id = base_state.get("session_id")

    async def _invoke(graph: CompiledStateGraph, sid: str) -> StoryState:
        # Create session-bound thread ID for scene processing
        tid = create_scene_thread_id(session_id, sid)
        session_manager.register_thread(session_id, tid)

        payload: StoryState = {
            **base_state,
            "scene_queue": [sid],
            "current_scene_id": sid,
            "thread_id": tid,
            "scene_states": {sid: {}},
        }

        # Ensure session_id is preserved in payload
        payload["session_id"] = session_id

        try:
            state = await graph.ainvoke(payload, {"configurable": {"thread_id": tid}})
            return cast(StoryState, state)
        finally:
            # Always unregister thread when done
            if session_id:
                session_manager.unregister_thread(session_id, tid)

    async with _get_checkpointer_context() as saver:
        graph = build_graph(checkpointer=saver)

        async def _sem_wrapper(sid: str) -> StoryState:
            if sem is None:
                return await _invoke(graph, sid)
            async with sem:
                return await _invoke(graph, sid)

        sem = asyncio.Semaphore(limit) if limit else None
        tasks = [asyncio.create_task(_sem_wrapper(sid)) for sid in scene_ids]
        states = await asyncio.gather(*tasks)
    return states


__all__ = ["create_story", "resume_story", "get_scene_status", "run_scene_threads"]
