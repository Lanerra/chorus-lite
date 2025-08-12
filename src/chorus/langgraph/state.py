# src/chorus/langgraph/state.py
"""State definition used during the LangGraph migration."""

from __future__ import annotations

import operator
import threading
from typing import Annotated, Any, TypedDict

from chorus.models import (
    CharacterProfile,
    Concept,
    SceneBrief,
    SceneStatus,
    Story,
    WorldAnvil,
)


class SessionContainer:
    """Session-scoped state container that provides complete isolation between sessions.

    Each session gets its own isolated state container to prevent cross-session
    contamination. The container tracks session-specific state and provides
    thread-safe access patterns.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize a new session container.

        Args:
            session_id: Unique identifier for this session
        """
        self.session_id = session_id
        self.state: dict[str, Any] = {}
        self.status: str = "initializing"
        self.active_threads: set[str] = set()
        self.lock = threading.RLock()

    def get_state(self) -> dict[str, Any]:
        """Get a copy of the current session state."""
        with self.lock:
            return self.state.copy()

    def update_state(self, updates: dict[str, Any]) -> None:
        """Update session state with new data."""
        with self.lock:
            self.state.update(updates)

    def set_status(self, status: str) -> None:
        """Update session status."""
        with self.lock:
            self.status = status

    def add_thread(self, thread_id: str) -> None:
        """Register an active thread for this session."""
        with self.lock:
            self.active_threads.add(thread_id)

    def remove_thread(self, thread_id: str) -> None:
        """Unregister a thread from this session."""
        with self.lock:
            self.active_threads.discard(thread_id)


class SessionManager:
    """Manages session-scoped state containers to eliminate cross-session contamination.

    Provides a singleton interface for creating and accessing session containers.
    Uses weak references to allow automatic cleanup of unused sessions.
    """

    _instance: SessionManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> SessionManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._sessions: dict[str, SessionContainer] = {}
            self._sessions_lock = threading.RLock()
            self._initialized = True

    def get_session_container(self, session_id: str) -> SessionContainer:
        """Get or create a session container for the given session ID.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionContainer for the specified session
        """
        with self._sessions_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionContainer(session_id)
            return self._sessions[session_id]

    def create_session(self, session_id: str) -> SessionContainer:
        """Create a new session container.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionContainer for the specified session
        """
        return self.get_session_container(session_id)

    async def get_or_create_session(self, session_id: str) -> SessionContainer:
        """Get or create a session container (async version).

        Args:
            session_id: Unique session identifier

        Returns:
            SessionContainer for the specified session
        """
        return self.get_session_container(session_id)

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Get current state for a session.

        Args:
            session_id: Session identifier

        Returns:
            Current session state dictionary
        """
        container = self.get_session_container(session_id)
        return container.get_state()

    def update_session_state(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update state for a specific session.

        Args:
            session_id: Session identifier
            updates: State updates to apply
        """
        container = self.get_session_container(session_id)
        container.update_state(updates)

    def get_session_status(self, session_id: str) -> str:
        """Get current status for a session.

        Args:
            session_id: Session identifier

        Returns:
            Current session status
        """
        container = self.get_session_container(session_id)
        return container.status

    def set_session_status(self, session_id: str, status: str) -> None:
        """Update status for a session.

        Args:
            session_id: Session identifier
            status: New status value
        """
        container = self.get_session_container(session_id)
        container.set_status(status)

    def register_thread(self, session_id: str, thread_id: str) -> None:
        """Register a thread as active for a session.

        Args:
            session_id: Session identifier
            thread_id: Thread identifier to register
        """
        container = self.get_session_container(session_id)
        container.add_thread(thread_id)

    def unregister_thread(self, session_id: str, thread_id: str) -> None:
        """Unregister a thread from a session.

        Args:
            session_id: Session identifier
            thread_id: Thread identifier to unregister
        """
        container = self.get_session_container(session_id)
        container.remove_thread(thread_id)

    def get_active_thread_count(self, session_id: str) -> int:
        """Get count of active threads for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of active threads
        """
        container = self.get_session_container(session_id)
        with container.lock:
            return len(container.active_threads)


class ContextSchema(TypedDict):
    """Shared thread-level context for the LangGraph pipeline.

    The context schema defines keys that persist across all nodes of a
    LangGraph execution. Now includes session-scoped isolation to prevent
    cross-session contamination.

    Attributes:
        thread_id: A unique string identifying the current thread or
            execution context.
        session_id: A unique string identifying the session to prevent
            cross-session state contamination.
    """

    thread_id: str
    session_id: str


class SceneState(TypedDict, total=False):
    """Per-scene progress tracking.

    Enhanced for single-pass workflow with coherence tracking.
    """

    draft: str | None
    # Allow multiple concurrent writes; reducer will keep the last non-empty value
    review_status: Annotated[list[str], operator.add]  # Legacy field for transition
    # New coherence tracking fields for single-pass workflow
    coherence_issues: Annotated[list[str], operator.add]
    coherence_passed: bool
    revision_notes: Annotated[list[str], operator.add]
    revision_history: Annotated[list[str], operator.add]
    feedback: Annotated[list[str], operator.add]
    lore_context: str | None
    lore_summary: str | None
    short_term_memory: Annotated[list[str], operator.add]
    memory_summary: str | None
    memory_context: str | None
    needs_revision: bool
    revision_count: int
    scene_status: SceneStatus
    draft_rejected: bool


class StoryState(TypedDict, total=False):
    """State container for the LangGraph workflow with session isolation.

    Enhanced with session-scoped state management to prevent cross-session
    contamination and support concurrent user workflows. Updated for single-pass
    workflow with coherence tracking fields.
    """

    # Session identification for isolation
    session_id: str
    idea: str
    concepts: list[Concept]
    vision: Concept | None
    outline: Story | None
    world_info: Annotated[list[WorldAnvil], operator.add]
    lore_entries: Annotated[list[WorldAnvil], operator.add]
    character_profiles: Annotated[list[CharacterProfile], operator.add]
    scene_briefs: Annotated[list[SceneBrief], operator.add]
    scene_queue: Annotated[list[str], operator.add]
    current_scene_id: str | None
    scene_states: Annotated[dict[str, SceneState], operator.or_]
    chapters: Annotated[list[str], operator.add]
    draft: str | None
    # Allow multiple concurrent writes; reducer will keep the last non-empty value
    review_status: Annotated[list[str], operator.add]  # Legacy field for transition
    # New coherence tracking fields for single-pass workflow
    coherence_issues: Annotated[list[str], operator.add]
    coherence_passed: bool
    revision_notes: Annotated[list[str], operator.add]
    revision_history: Annotated[list[str], operator.add]
    feedback: Annotated[list[str], operator.add]
    lore_context: str | None
    lore_summary: str | None
    character_clusters: Annotated[list[list[str]], operator.add]
    relationship_triples: Annotated[list, operator.add]
    discoveries: Annotated[list, operator.add]
    short_term_memory: Annotated[list[str], operator.add]
    memory_summary: str | None
    memory_context: str | None
    thread_id: str | None
    needs_revision: bool
    revision_count: int
    scene_status: SceneStatus
    concept_rejected: bool
    draft_rejected: bool
    # Allow multiple error messages per step; LangGraph will aggregate using list-add
    error_messages: Annotated[list[str], operator.add]
    # Track nodes that failed within a single tick; supports multiple concurrent failures
    error_nodes: Annotated[list[str], operator.add]


__all__ = [
    "StoryState",
    "SceneState",
    "SessionManager",
    "SessionContainer",
    "ContextSchema",
]
