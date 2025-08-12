# src/chorus/langgraph/session_utils.py
"""Session management utilities for thread ID extraction and validation."""

from __future__ import annotations

import re
import uuid

from chorus.core.logs import get_logger

logger = get_logger(__name__)


def extract_session_id_from_thread_id(thread_id: str) -> str | None:
    """Extract session ID from session-bound thread ID format.

    Thread ID format: {session_id}_{uuid}
    UUID format: 8-4-4-4-12 hex digits with dashes

    This function handles session IDs that contain underscores by looking for
    the UUID pattern and extracting everything before the last UUID.

    Args:
        thread_id: The thread ID to extract session ID from

    Returns:
        The extracted session ID, or None if extraction fails

    Examples:
        >>> extract_session_id_from_thread_id("session-123_a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        'session-123'
        >>> extract_session_id_from_thread_id("user_abc_def_a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        'user_abc_def'
        >>> extract_session_id_from_thread_id("no_uuid_here")
        'no'  # Falls back to naive split
    """
    if "_" not in thread_id:
        return None

    # UUID pattern: 8-4-4-4-12 hex digits
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

    # Find all UUID matches in the thread_id
    uuid_matches = list(re.finditer(uuid_pattern, thread_id, re.IGNORECASE))

    if not uuid_matches:
        # No UUID found, use naive split
        logger.debug(
            "No UUID pattern found in thread_id '%s', using naive split", thread_id
        )
        return thread_id.split("_")[0]

    # Get the position of the last UUID
    last_uuid = uuid_matches[-1]
    uuid_start = last_uuid.start()

    # Find the underscore immediately before the UUID
    underscore_pos = thread_id.rfind("_", 0, uuid_start)
    if underscore_pos == -1:
        logger.warning("No underscore found before UUID in thread_id '%s'", thread_id)
        return None

    # Extract everything before the underscore
    session_id = thread_id[:underscore_pos]
    logger.debug("Extracted session_id '%s' from thread_id '%s'", session_id, thread_id)
    return session_id


def create_session_bound_thread_id(session_id: str | None) -> str:
    """Create a session-bound thread ID in the standard format.

    Args:
        session_id: The session identifier (optional)

    Returns:
        A thread ID in the format {session_id}_{uuid} or {uuid} if session_id is None

    Examples:
        >>> create_session_bound_thread_id("session-123")
        'session-123_a1b2c3d4-e5f6-7890-abcd-ef1234567890'
        >>> create_session_bound_thread_id(None)
        'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    """
    thread_uuid = str(uuid.uuid4())
    if session_id is None:
        thread_id = thread_uuid
        logger.debug(
            "Created thread_id '%s' (no session_id)", thread_id
        )
    else:
        thread_id = f"{session_id}_{thread_uuid}"
        logger.debug(
            "Created session-bound thread_id '%s' for session '%s'", thread_id, session_id
        )
    return thread_id


def validate_session_id(session_id: str | None) -> bool:
    """Validate that a session ID is valid and non-empty.

    Args:
        session_id: The session ID to validate

    Returns:
        True if the session ID is valid, False otherwise
    """
    if not session_id:
        return False

    if not isinstance(session_id, str):
        return False

    # Session ID should not be just whitespace
    if not session_id.strip():
        return False

    # Reasonable length limits (prevent extremely long session IDs)
    if len(session_id) > 500:
        logger.warning(
            "Session ID is unusually long (%d chars): %s",
            len(session_id),
            session_id[:50] + "...",
        )
        return False

    return True


def validate_thread_id_format(thread_id: str) -> bool:
    """Validate that a thread ID follows the session-bound format.

    Args:
        thread_id: The thread ID to validate

    Returns:
        True if the thread ID appears to be session-bound, False otherwise
    """
    if not thread_id or not isinstance(thread_id, str):
        return False

    # Should contain at least one underscore
    if "_" not in thread_id:
        return False

    # Should end with a UUID pattern
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    if not re.search(uuid_pattern, thread_id, re.IGNORECASE):
        return False

    # Should be able to extract a session ID
    session_id = extract_session_id_from_thread_id(thread_id)
    return validate_session_id(session_id)


def create_scene_thread_id(session_id: str, scene_id: str) -> str:
    """Create a scene-specific thread ID for concurrent scene processing.

    Args:
        session_id: The session identifier
        scene_id: The scene identifier

    Returns:
        A scene-specific thread ID in the format {session_id}_scene_{scene_id}_{uuid}
    """
    thread_uuid = str(uuid.uuid4())
    thread_id = f"{session_id}_scene_{scene_id}_{thread_uuid}"
    logger.debug(
        "Created scene thread_id '%s' for session '%s' scene '%s'",
        thread_id,
        session_id,
        scene_id,
    )
    return thread_id


__all__ = [
    "extract_session_id_from_thread_id",
    "create_session_bound_thread_id",
    "validate_session_id",
    "validate_thread_id_format",
    "create_scene_thread_id",
]
