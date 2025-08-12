# src/chorus/langgraph/versioning.py
"""Checkpoint versioning and conflict resolution for distributed LangGraph coordination."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chorus.core.logs import get_logger

logger = get_logger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving checkpoint conflicts."""

    LATEST_WINS = "latest_wins"  # Last write wins
    MERGE_STATES = "merge_states"  # Attempt to merge non-conflicting changes
    MANUAL_RESOLUTION = "manual_resolution"  # Require manual intervention
    ABORT_ON_CONFLICT = "abort_on_conflict"  # Fail and require retry


@dataclass
class CheckpointVersion:
    """Version information for a checkpoint."""

    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_number: int = 0
    timestamp: float = field(default_factory=time.time)
    parent_version: str | None = None
    branch_id: str = "main"
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_descendant_of(self, other_version: CheckpointVersion) -> bool:
        """Check if this version is a descendant of another version."""
        return self.sequence_number > other_version.sequence_number and (
            self.parent_version == other_version.version_id
            or self.branch_id == other_version.branch_id
        )

    def is_concurrent_with(self, other_version: CheckpointVersion) -> bool:
        """Check if this version is concurrent (conflicting) with another."""
        return (
            self.sequence_number == other_version.sequence_number
            and self.version_id != other_version.version_id
            and self.parent_version == other_version.parent_version
        )


@dataclass
class ConflictInfo:
    """Information about a detected checkpoint conflict."""

    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    current_version: CheckpointVersion = field(default_factory=CheckpointVersion)
    incoming_version: CheckpointVersion = field(default_factory=CheckpointVersion)
    conflict_type: str = "concurrent_update"
    conflicting_fields: list[str] = field(default_factory=list)
    resolution_strategy: ConflictResolutionStrategy = (
        ConflictResolutionStrategy.LATEST_WINS
    )
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_result: dict[str, Any] | None = None


class CheckpointVersionManager:
    """Manages checkpoint versioning and conflict resolution."""

    def __init__(
        self,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_STATES,
    ):
        self.default_strategy = default_strategy
        self._version_history: dict[str, list[CheckpointVersion]] = {}
        self._active_conflicts: dict[str, ConflictInfo] = {}

    def create_version(
        self,
        session_id: str,
        checkpoint_data: Any,
        parent_version: str | None = None,
        branch_id: str = "main",
    ) -> CheckpointVersion:
        """Create a new checkpoint version."""

        # Get the current sequence number for this session
        history = self._version_history.get(session_id, [])
        sequence_number = len(history)

        # Create version with checksum for conflict detection
        version = CheckpointVersion(
            sequence_number=sequence_number,
            parent_version=parent_version,
            branch_id=branch_id,
            checksum=self._calculate_checksum(checkpoint_data),
            metadata={
                "session_id": session_id,
                "data_size": len(str(checkpoint_data)),
                "created_at": time.time(),
            },
        )

        # Add to history
        if session_id not in self._version_history:
            self._version_history[session_id] = []
        self._version_history[session_id].append(version)

        logger.debug(
            f"Created checkpoint version {version.version_id} for session {session_id}"
        )
        return version

    def detect_conflict(
        self,
        session_id: str,
        incoming_version: CheckpointVersion,
        current_data: Any,
        incoming_data: Any,
    ) -> ConflictInfo | None:
        """Detect conflicts between checkpoint versions."""

        history = self._version_history.get(session_id, [])
        if not history:
            return None  # No conflict for first checkpoint

        current_version = history[-1]  # Latest version

        # Check for concurrent updates
        if incoming_version.is_concurrent_with(current_version):
            conflict = ConflictInfo(
                session_id=session_id,
                current_version=current_version,
                incoming_version=incoming_version,
                conflict_type="concurrent_update",
                conflicting_fields=self._identify_conflicting_fields(
                    current_data, incoming_data
                ),
                resolution_strategy=self.default_strategy,
            )

            self._active_conflicts[conflict.conflict_id] = conflict
            logger.warning(
                f"Detected checkpoint conflict {conflict.conflict_id} for session {session_id}"
            )
            return conflict

        # Check for out-of-order updates
        if incoming_version.sequence_number < current_version.sequence_number:
            conflict = ConflictInfo(
                session_id=session_id,
                current_version=current_version,
                incoming_version=incoming_version,
                conflict_type="out_of_order_update",
                resolution_strategy=ConflictResolutionStrategy.ABORT_ON_CONFLICT,
            )

            self._active_conflicts[conflict.conflict_id] = conflict
            logger.warning(
                f"Detected out-of-order update {conflict.conflict_id} for session {session_id}"
            )
            return conflict

        return None

    def resolve_conflict(
        self, conflict: ConflictInfo, current_data: Any, incoming_data: Any
    ) -> dict[str, Any]:
        """Resolve a checkpoint conflict based on the resolution strategy."""

        try:
            if conflict.resolution_strategy == ConflictResolutionStrategy.LATEST_WINS:
                result = self._resolve_latest_wins(
                    conflict, current_data, incoming_data
                )

            elif (
                conflict.resolution_strategy == ConflictResolutionStrategy.MERGE_STATES
            ):
                result = self._resolve_merge_states(
                    conflict, current_data, incoming_data
                )

            elif (
                conflict.resolution_strategy
                == ConflictResolutionStrategy.ABORT_ON_CONFLICT
            ):
                raise ConflictResolutionError(
                    f"Conflict {conflict.conflict_id} requires manual resolution"
                )

            else:  # MANUAL_RESOLUTION
                raise ConflictResolutionError(
                    f"Conflict {conflict.conflict_id} requires manual intervention"
                )

            # Mark conflict as resolved
            conflict.resolved = True
            conflict.resolution_result = result

            logger.info(
                f"Resolved conflict {conflict.conflict_id} using {conflict.resolution_strategy.value}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict.conflict_id}: {e}")
            raise

    def _resolve_latest_wins(
        self, conflict: ConflictInfo, current_data: Any, incoming_data: Any
    ) -> dict[str, Any]:
        """Resolve conflict by using the latest (incoming) data."""
        return {
            "resolved_data": incoming_data,
            "strategy": "latest_wins",
            "winner": "incoming",
            "version": conflict.incoming_version.version_id,
        }

    def _resolve_merge_states(
        self, conflict: ConflictInfo, current_data: Any, incoming_data: Any
    ) -> dict[str, Any]:
        """Resolve conflict by attempting to merge non-conflicting changes."""

        if not isinstance(current_data, dict) or not isinstance(incoming_data, dict):
            # Fallback to latest wins for non-dict data
            return self._resolve_latest_wins(conflict, current_data, incoming_data)

        merged_data = {}
        conflicting_keys = []

        # Get all keys from both datasets
        all_keys = set(current_data.keys()) | set(incoming_data.keys())

        for key in all_keys:
            current_value = current_data.get(key)
            incoming_value = incoming_data.get(key)

            if key not in current_data:
                # New key in incoming data
                merged_data[key] = incoming_value
            elif key not in incoming_data:
                # Key removed in incoming data (keep current)
                merged_data[key] = current_value
            elif current_value == incoming_value:
                # No conflict - same value
                merged_data[key] = current_value
            else:
                # Conflict detected
                conflicting_keys.append(key)
                # Use incoming value for conflicts (latest wins for individual fields)
                merged_data[key] = incoming_value

        return {
            "resolved_data": merged_data,
            "strategy": "merge_states",
            "conflicting_keys": conflicting_keys,
            "version": f"merged_{conflict.incoming_version.version_id}",
        }

    def _identify_conflicting_fields(
        self, current_data: Any, incoming_data: Any
    ) -> list[str]:
        """Identify fields that have conflicting values."""

        if not isinstance(current_data, dict) or not isinstance(incoming_data, dict):
            return ["data"]  # Treat entire data as conflicting for non-dict types

        conflicting_fields = []
        all_keys = set(current_data.keys()) | set(incoming_data.keys())

        for key in all_keys:
            current_value = current_data.get(key)
            incoming_value = incoming_data.get(key)

            if current_value != incoming_value:
                conflicting_fields.append(key)

        return conflicting_fields

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate a simple checksum for conflict detection."""
        import hashlib

        data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def get_version_history(self, session_id: str) -> list[CheckpointVersion]:
        """Get version history for a session."""
        return self._version_history.get(session_id, [])

    def get_active_conflicts(self, session_id: str | None = None) -> list[ConflictInfo]:
        """Get active conflicts, optionally filtered by session."""
        conflicts = list(self._active_conflicts.values())

        if session_id:
            conflicts = [
                c for c in conflicts if c.session_id == session_id and not c.resolved
            ]

        return conflicts

    def cleanup_resolved_conflicts(self, older_than_hours: int = 24) -> int:
        """Clean up resolved conflicts older than specified hours."""
        cutoff_time = time.time() - (older_than_hours * 3600)

        resolved_conflicts = [
            conflict_id
            for conflict_id, conflict in self._active_conflicts.items()
            if conflict.resolved and conflict.detected_at < cutoff_time
        ]

        for conflict_id in resolved_conflicts:
            del self._active_conflicts[conflict_id]

        logger.info(f"Cleaned up {len(resolved_conflicts)} resolved conflicts")
        return len(resolved_conflicts)


class ConflictResolutionError(Exception):
    """Exception raised when conflict resolution fails."""

    pass


# Integration with the enhanced checkpointer
class VersionedCheckpointMixin:
    """Mixin to add versioning capabilities to checkpoint savers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version_manager = CheckpointVersionManager()

    async def versioned_put(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_STATES,
    ) -> Any:
        """Put checkpoint with versioning and conflict resolution."""

        session_id = self._extract_session_id(config)

        # Create version for incoming checkpoint
        incoming_version = self.version_manager.create_version(
            session_id, checkpoint.data if hasattr(checkpoint, "data") else checkpoint
        )

        # Check for existing checkpoint to detect conflicts
        try:
            current_checkpoint = await self.get(config)
            if current_checkpoint:
                current_data = (
                    current_checkpoint.data
                    if hasattr(current_checkpoint, "data")
                    else current_checkpoint
                )
                incoming_data = (
                    checkpoint.data if hasattr(checkpoint, "data") else checkpoint
                )

                # Detect potential conflicts
                conflict = self.version_manager.detect_conflict(
                    session_id, incoming_version, current_data, incoming_data
                )

                if conflict:
                    # Resolve the conflict
                    conflict.resolution_strategy = strategy
                    resolution_result = self.version_manager.resolve_conflict(
                        conflict, current_data, incoming_data
                    )

                    # Update checkpoint with resolved data
                    if hasattr(checkpoint, "data"):
                        checkpoint.data = resolution_result["resolved_data"]
                    else:
                        checkpoint = resolution_result["resolved_data"]

                    logger.info(
                        f"Resolved checkpoint conflict for session {session_id}"
                    )

        except Exception as e:
            logger.debug(f"No existing checkpoint found for conflict detection: {e}")

        # Add version metadata to checkpoint
        if hasattr(checkpoint, "data") and isinstance(checkpoint.data, dict):
            checkpoint.data["_version"] = {
                "version_id": incoming_version.version_id,
                "sequence_number": incoming_version.sequence_number,
                "timestamp": incoming_version.timestamp,
                "checksum": incoming_version.checksum,
            }

        # Proceed with normal put operation
        return await self.put(config, checkpoint, metadata)

    def get_version_info(self, session_id: str) -> dict[str, Any]:
        """Get versioning information for a session."""
        history = self.version_manager.get_version_history(session_id)
        conflicts = self.version_manager.get_active_conflicts(session_id)

        return {
            "session_id": session_id,
            "version_count": len(history),
            "latest_version": history[-1].__dict__ if history else None,
            "active_conflicts": len(conflicts),
            "conflict_details": [c.__dict__ for c in conflicts],
        }


__all__ = [
    "CheckpointVersion",
    "ConflictInfo",
    "CheckpointVersionManager",
    "ConflictResolutionStrategy",
    "ConflictResolutionError",
    "VersionedCheckpointMixin",
]
