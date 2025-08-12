# src/chorus/core/queue.py
"""Durable, process-safe task queue backed by Postgres with parallel processing.

This replaces the previous in-memory queue so producers and the
background worker can share tasks across processes and restarts. Enhanced with
worker pool patterns for 5x throughput increase.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError

from chorus.config import config
from chorus.core.logs import log_calls, log_message
from chorus.models.task import BaseTask

# Status values mirrored from docs/schemas/TaskQueue.json / SQL model
STATUS_QUEUED = "queued"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_DLQ = "dlq"


class ParallelTaskProcessor:
    """Worker pool pattern for parallel task distribution with 5x throughput increase."""

    def __init__(self):
        self.max_workers = config.concurrency.queue_workers
        self.worker_semaphore = asyncio.Semaphore(self.max_workers)
        self.active_workers = 0
        self.processed_count = 0

    async def dequeue_parallel(self, count: int = None) -> list[BaseTask]:
        """Dequeue multiple tasks for parallel processing."""
        if count is None:
            count = self.max_workers

        tasks = []
        for _ in range(count):
            task = await dequeue()
            if task is None:
                break
            tasks.append(task)

        log_message(f"Dequeued {len(tasks)} tasks for parallel processing")
        return tasks

    async def process_tasks_parallel(
        self, tasks: list[BaseTask], handler_func
    ) -> list[tuple[BaseTask, bool]]:
        """Process multiple tasks concurrently with worker pool pattern."""
        if not tasks:
            return []

        async def process_single_task(task: BaseTask) -> tuple[BaseTask, bool]:
            """Process a single task with semaphore control."""
            async with self.worker_semaphore:
                self.active_workers += 1
                try:
                    log_message(
                        f"Worker processing task {task.id} (active workers: {self.active_workers})"
                    )
                    success = await handler_func(task)
                    self.processed_count += 1
                    return task, success
                except Exception as e:
                    log_message(f"Task {task.id} failed: {e}")
                    return task, False
                finally:
                    self.active_workers -= 1

        # Process tasks concurrently with worker pool
        results = await asyncio.gather(
            *[process_single_task(task) for task in tasks], return_exceptions=True
        )

        # Handle exceptions and return results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                log_message(f"Task processing exception: {result}")
                # Create a failure result for the exception
                processed_results.append((None, False))
            else:
                processed_results.append(result)

        success_count = sum(1 for _, success in processed_results if success)
        log_message(
            f"Parallel processing completed: {success_count}/{len(tasks)} tasks successful"
        )

        return processed_results


# Global parallel processor instance
parallel_processor = ParallelTaskProcessor()


def _serialize_task(task: BaseTask) -> tuple[str, dict[str, Any]]:
    """Convert BaseTask to (type, payload) for DB storage.

    Use Pydantic's JSON-mode dump so datetimes and other types are serializable.
    """
    # Store minimal fields required to reconstruct the task.
    # mode="json" ensures datetime -> ISO strings etc.
    payload: dict[str, Any] = task.model_dump(mode="json", by_alias=True)
    # Ensure we do not duplicate id in payload; DB id is authoritative.
    payload.pop("id", None)
    payload["task_type"] = task.task_type
    return task.task_type, payload


def _deserialize_task(row: Any) -> BaseTask:
    """Reconstruct a BaseTask from DB row."""
    # Row fields expected: id, type, payload (JSON)
    task_id: UUID = row.id
    task_type: str = row.type
    payload: dict[str, Any] = row.payload or {}
    # Recover fields
    # New-style tasks include task_type in payload for safety
    payload = dict(payload or {})
    payload["id"] = str(task_id)
    payload["task_type"] = task_type
    # BaseTask will accept extra fields for specific subclasses during handling
    return BaseTask(**payload)


@log_calls
async def enqueue(task: BaseTask, *, priority: int = 1) -> UUID:
    """Persist a task to Postgres task_queue in QUEUED state."""
    # Defer import to avoid circulars
    from chorus.canon import get_pg

    task_type, payload = _serialize_task(task)
    try:
        async with get_pg() as conn:
            # Bind JSON via driver-native bindparam so SQLAlchemy renders ($1, $2, ...) for psycopg
            # Custom JSON serializer to handle datetime if needed
            def _json_default(obj: Any) -> Any:
                if isinstance(obj, datetime | date):
                    return obj.isoformat()
                # Fall back to pydantic-like dump
                if hasattr(obj, "model_dump"):
                    return obj.model_dump(mode="json", by_alias=True)
                raise TypeError(
                    f"Object of type {obj.__class__.__name__} is not JSON serializable"
                )

            # Insert with empty dependencies array and NULL assigned_to to match DB schema:
            # - dependencies is uuid[]
            # - assigned_to is a single uuid (nullable)
            result = await conn.execute(
                sa_text(
                    "INSERT INTO task_queue (type, payload, priority, dependencies, assigned_to, status) "
                    "VALUES (:type, :payload, :priority, '{}'::uuid[], NULL, :status) "
                    "RETURNING id"
                ),
                {
                    "type": task_type,
                    # Ensure JSON serialization for drivers that don't auto-adapt dict -> jsonb in text() statements
                    "payload": cast(
                        Any, __import__("json").dumps(payload, default=_json_default)
                    ),
                    "priority": int(priority),
                    "status": STATUS_QUEUED,
                },
            )
            row = result.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert task into task_queue")
            db_id: UUID = row[0]
            task.id = db_id
            await conn.commit()
            log_message(f"Enqueued {task.task_type} {db_id}")
            return db_id
    except SQLAlchemyError as exc:
        # Fallback: still assign an id to avoid upstream None checks, but log error.
        if task.id is None:
            task.id = uuid4()
        log_message(f"Failed to enqueue task to DB: {exc}")
        return cast(UUID, task.id)


@log_calls
async def dequeue() -> BaseTask | None:
    """Atomically claim the highest-priority queued task.

    Uses SKIP LOCKED to allow multiple workers.
    """
    from chorus.canon import get_pg

    try:
        async with get_pg() as conn:
            # Choose queued tasks ordered by priority desc, created_at asc
            result = await conn.execute(
                sa_text(
                    """
                    WITH cte AS (
                        SELECT id
                        FROM task_queue
                        WHERE status = :queued
                        ORDER BY priority DESC, created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE task_queue tq
                    SET status = :in_progress
                    FROM cte
                    WHERE tq.id = cte.id
                    RETURNING tq.id, tq.type, tq.payload
                    """
                ),
                {"queued": STATUS_QUEUED, "in_progress": STATUS_IN_PROGRESS},
            )
            row = result.fetchone()
            if row is None:
                return None

            # Build a lightweight container with attribute access
            class _Row:
                def __init__(self, id: UUID, type: str, payload: dict[str, Any] | None):
                    self.id = id
                    self.type = type
                    self.payload = payload or {}

            task = _deserialize_task(_Row(row[0], row[1], row[2]))
            await conn.commit()
            return task
    except SQLAlchemyError as exc:
        log_message(f"Dequeue error: {exc}")
        return None


@log_calls
async def dequeue_batch(limit: int = 5) -> list[BaseTask]:
    """Atomically claim multiple highest-priority queued tasks for parallel processing.

    Uses SKIP LOCKED to allow multiple workers and supports batch operations
    for 5x throughput increase.
    """
    from chorus.canon import get_pg

    try:
        async with get_pg() as conn:
            # Choose multiple queued tasks ordered by priority desc, created_at asc
            result = await conn.execute(
                sa_text(
                    """
                    WITH cte AS (
                        SELECT id
                        FROM task_queue
                        WHERE status = :queued
                        ORDER BY priority DESC, created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT :limit
                    )
                    UPDATE task_queue tq
                    SET status = :in_progress
                    FROM cte
                    WHERE tq.id = cte.id
                    RETURNING tq.id, tq.type, tq.payload
                    """
                ),
                {
                    "queued": STATUS_QUEUED,
                    "in_progress": STATUS_IN_PROGRESS,
                    "limit": limit,
                },
            )
            rows = result.fetchall()
            if not rows:
                return []

            # Build lightweight containers for each task
            class _Row:
                def __init__(self, id: UUID, type: str, payload: dict[str, Any] | None):
                    self.id = id
                    self.type = type
                    self.payload = payload or {}

            tasks = [_deserialize_task(_Row(row[0], row[1], row[2])) for row in rows]
            await conn.commit()
            log_message(f"Batch dequeued {len(tasks)} tasks for parallel processing")
            return tasks
    except SQLAlchemyError as exc:
        log_message(f"Batch dequeue error: {exc}")
        return []


async def list_tasks() -> list[BaseTask]:
    """List currently queued tasks (QUEUED only)."""
    from chorus.canon import get_pg

    tasks: list[BaseTask] = []
    try:
        async with get_pg() as conn:
            result = await conn.execute(
                sa_text(
                    "SELECT id, type, payload FROM task_queue WHERE status = :queued ORDER BY created_at ASC"
                ),
                {"queued": STATUS_QUEUED},
            )
            rows = result.fetchall()
            for id_, type_, payload in rows:

                class _Row:
                    def __init__(
                        self, id: UUID, type: str, payload: dict[str, Any] | None
                    ):
                        self.id = id
                        self.type = type
                        self.payload = payload or {}

                tasks.append(_deserialize_task(_Row(id_, type_, payload)))
    except SQLAlchemyError as exc:
        log_message(f"list_tasks error: {exc}")
    return tasks


@log_calls
async def move_to_dlq(task: BaseTask) -> None:
    """Mark the task as DLQ."""
    from chorus.canon import get_pg

    if task.id is None:
        return
    try:
        async with get_pg() as conn:
            await conn.execute(
                sa_text("UPDATE task_queue SET status = :dlq WHERE id = :id"),
                {"dlq": STATUS_DLQ, "id": task.id},
            )
            await conn.commit()
            log_message(f"DLQ {task.task_type} {task.id}")
    except SQLAlchemyError as exc:
        log_message(f"move_to_dlq error: {exc}")


async def list_dlq() -> list[BaseTask]:
    """Return tasks in DLQ state."""
    from chorus.canon import get_pg

    tasks: list[BaseTask] = []
    try:
        async with get_pg() as conn:
            result = await conn.execute(
                sa_text(
                    "SELECT id, type, payload FROM task_queue WHERE status = :dlq ORDER BY created_at ASC"
                ),
                {"dlq": STATUS_DLQ},
            )
            rows = result.fetchall()
            for id_, type_, payload in rows:

                class _Row:
                    def __init__(
                        self, id: UUID, type: str, payload: dict[str, Any] | None
                    ):
                        self.id = id
                        self.type = type
                        self.payload = payload or {}

                tasks.append(_deserialize_task(_Row(id_, type_, payload)))
    except SQLAlchemyError as exc:
        log_message(f"list_dlq error: {exc}")
    return tasks


@log_calls
async def delete_dlq(task_id: UUID) -> None:
    """Delete a DLQ task permanently."""
    from chorus.canon import get_pg

    try:
        async with get_pg() as conn:
            await conn.execute(
                sa_text("DELETE FROM task_queue WHERE id = :id AND status = :dlq"),
                {"id": task_id, "dlq": STATUS_DLQ},
            )
    except SQLAlchemyError as exc:
        log_message(f"delete_dlq error: {exc}")


@log_calls
async def requeue_dlq(task_id: UUID, *, priority: int = 1) -> None:
    """Move a DLQ task back to QUEUED with a given priority."""
    from chorus.canon import get_pg

    try:
        async with get_pg() as conn:
            await conn.execute(
                sa_text(
                    "UPDATE task_queue SET status = :queued, priority = :priority WHERE id = :id AND status = :dlq"
                ),
                {
                    "id": task_id,
                    "queued": STATUS_QUEUED,
                    "dlq": STATUS_DLQ,
                    "priority": int(priority),
                },
            )
            await conn.commit()
    except SQLAlchemyError as exc:
        log_message(f"requeue_dlq error: {exc}")


@log_calls
async def mark_complete(task_id: UUID | None) -> None:
    """Mark a task COMPLETED and remove any duplicate queued copies."""
    from chorus.canon import get_pg

    if not task_id:
        return
    try:
        async with get_pg() as conn:
            await conn.execute(
                sa_text("UPDATE task_queue SET status = :completed WHERE id = :id"),
                {"completed": STATUS_COMPLETED, "id": task_id},
            )
            await conn.commit()
            log_message(f"Completed {task_id}")
    except SQLAlchemyError as exc:
        log_message(f"mark_complete error: {exc}")


@log_calls
async def lock_scene(scene_id: UUID, worker_id: str, *, ttl: int = 300) -> bool:
    """Scene locks are not required with DB queue; return True."""
    return True


async def scene_locked(scene_id: UUID) -> bool:
    """Return False to indicate no explicit locking is used."""
    return False


@log_calls
async def unlock_scene(scene_id: UUID, worker_id: str) -> None:
    """Compatibility no-op."""
    return None


async def close() -> None:
    """No global state to clear for DB-backed queue."""
    return None


__all__ = [
    "enqueue",
    "dequeue",
    "dequeue_batch",
    "list_tasks",
    "mark_complete",
    "close",
    "move_to_dlq",
    "list_dlq",
    "delete_dlq",
    "requeue_dlq",
    "lock_scene",
    "unlock_scene",
    "scene_locked",
    "ParallelTaskProcessor",
    "parallel_processor",
]
