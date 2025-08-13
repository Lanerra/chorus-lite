# src/chorus/canon/db.py
"""Database session creation, migrations, and helpers."""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command as alembic_command  # type: ignore[attr-defined]
from alembic.config import Config
from sqlalchemy import BigInteger, bindparam
from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import func, select

from chorus.config import config
from chorus.core.logs import log_message, get_event_logger, EventType, Priority

# Initialize EventLogger for database session management
event_logger = get_event_logger()

DATABASE_URL = config.database.postgres_url

_ENGINE = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(
    bind=_ENGINE, class_=AsyncSession, expire_on_commit=False
)


async def _maybe_await(value: Any) -> Any:
    """Return awaited ``value`` if it is awaitable."""

    if inspect.isawaitable(value):
        return await value
    return value


@asynccontextmanager
async def get_pg() -> AsyncIterator[AsyncSession]:
    """Return a SQLAlchemy asynchronous session."""
    
    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        "Creating PostgreSQL session",
        Priority.NORMAL,
        metadata={"operation": "session_create", "database": "postgres"}
    )
    
    try:
        async with SessionLocal() as session:
            creation_time = time.time() - start_time
            await event_logger.log(
                EventType.DATABASE_OPERATION,
                "PostgreSQL session created successfully",
                Priority.NORMAL,
                metadata={
                    "operation": "session_create",
                    "database": "postgres",
                    "creation_time": creation_time,
                    "success": True
                }
            )
            yield session
            
            total_time = time.time() - start_time
            await event_logger.log(
                EventType.DATABASE_OPERATION,
                "PostgreSQL session completed",
                Priority.NORMAL,
                metadata={
                    "operation": "session_complete",
                    "database": "postgres",
                    "total_duration": total_time
                }
            )
    except SQLAlchemyError as exc:  # pragma: no cover - connection errors
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(exc).__name__,
            error_msg=str(exc),
            context="PostgreSQL session creation",
            metadata={
                "operation": "session_create",
                "database": "postgres",
                "duration": duration,
                "error_category": "connection"
            }
        )
        log_message(f"PostgreSQL connection error: {exc}")
        raise


@asynccontextmanager
async def advisory_lock(session: AsyncSession, lock_id: int) -> AsyncIterator[None]:
    """Acquire a PostgreSQL advisory lock.

    Uses explicit bigint casts to avoid psycopg/SQLAlchemy binding as NUMERIC.
    """

    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        f"Acquiring PostgreSQL advisory lock: {lock_id}",
        Priority.NORMAL,
        metadata={"operation": "advisory_lock_acquire", "lock_id": lock_id}
    )

    # Use typed bindparam (BIGINT) to satisfy pg_advisory_lock signature with psycopg3
    stmt_lock = select(func.pg_advisory_lock(bindparam("id", type_=BigInteger))).params(
        id=int(lock_id)
    )
    
    try:
        await session.execute(stmt_lock)
        lock_time = time.time() - start_time
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Successfully acquired advisory lock: {lock_id}",
            Priority.NORMAL,
            metadata={
                "operation": "advisory_lock_acquire",
                "lock_id": lock_id,
                "lock_time": lock_time,
                "success": True
            }
        )
        
        try:
            yield
        finally:
            await event_logger.log(
                EventType.DATABASE_OPERATION,
                f"Releasing PostgreSQL advisory lock: {lock_id}",
                Priority.NORMAL,
                metadata={"operation": "advisory_lock_release", "lock_id": lock_id}
            )
            
            stmt_unlock = select(
                func.pg_advisory_unlock(bindparam("id", type_=BigInteger))
            ).params(id=int(lock_id))
            await session.execute(stmt_unlock)
            
            total_time = time.time() - start_time
            await event_logger.log(
                EventType.DATABASE_OPERATION,
                f"Successfully released advisory lock: {lock_id}",
                Priority.NORMAL,
                metadata={
                    "operation": "advisory_lock_release",
                    "lock_id": lock_id,
                    "total_duration": total_time,
                    "success": True
                }
            )
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"Advisory lock operations for lock_id {lock_id}",
            metadata={
                "operation": "advisory_lock",
                "lock_id": lock_id,
                "duration": duration
            }
        )
        raise


async def ensure_schema() -> None:
    """Initialize extensions and apply migrations once using an advisory lock.

    Prevents infinite re-entry when multiple startup paths call ensure_schema()
    (e.g., Uvicorn reloaders, multiple nodes). Uses a session-level advisory
    lock and stamps baseline if DB is empty before upgrading to heads.
    """
    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        "Starting database schema initialization",
        Priority.HIGH,
        metadata={"operation": "schema_ensure", "phase": "start"}
    )
    
    # Use a stable advisory lock id (signed 64-bit). Hash of 'chorus_schema_lock'.
    # Truncate to 63 bits to ensure it fits into BIGINT and retains sign safety.
    _RAW_LOCK_ID = 0x43686F7275735F534348454D41
    LOCK_ID = int(_RAW_LOCK_ID & 0x7FFF_FFFF_FFFF_FFFF)

    try:
        async with get_pg() as session:
            await event_logger.log(
                EventType.DATABASE_OPERATION,
                f"Acquiring schema lock: {LOCK_ID}",
                Priority.HIGH,
                metadata={"operation": "schema_ensure", "phase": "lock_acquire", "lock_id": LOCK_ID}
            )
            
            # Acquire advisory lock to serialize schema setup across processes
            stmt_lock = select(
                func.pg_advisory_lock(bindparam("id", type_=BigInteger))
            ).params(id=LOCK_ID)
            await session.execute(stmt_lock)
            
            try:
                await event_logger.log(
                    EventType.DATABASE_OPERATION,
                    "Checking for existing alembic version table",
                    Priority.HIGH,
                    metadata={"operation": "schema_ensure", "phase": "check_alembic"}
                )
                
                # If any admin table exists, assume migrations have been applied sufficiently.
                result = await session.execute(
                    sa_text("SELECT to_regclass('public.alembic_version')")
                )
                alembic_present = await _maybe_await(result.scalar())

                # If alembic_version is missing, stamp baseline then upgrade heads
                root = Path(__file__).resolve().parents[3]
                cfg = Config(str(root / "alembic.ini"))

                if alembic_present is None:
                    await event_logger.log(
                        EventType.DATABASE_OPERATION,
                        "Alembic version table not found, stamping baseline",
                        Priority.HIGH,
                        metadata={"operation": "schema_ensure", "phase": "stamp_baseline", "revision": "20250805_baseline"}
                    )
                    # Stamp the initial baseline revision without running it repeatedly
                    alembic_command.stamp(cfg, "20250805_baseline")  # type: ignore[attr-defined]
                else:
                    await event_logger.log(
                        EventType.DATABASE_OPERATION,
                        "Alembic version table found, skipping baseline stamp",
                        Priority.HIGH,
                        metadata={"operation": "schema_ensure", "phase": "baseline_exists"}
                    )

                await event_logger.log(
                    EventType.DATABASE_OPERATION,
                    "Upgrading database to latest heads",
                    Priority.HIGH,
                    metadata={"operation": "schema_ensure", "phase": "upgrade_heads"}
                )
                
                # Now upgrade to latest heads exactly once
                alembic_command.upgrade(cfg, "heads")  # type: ignore[attr-defined]
                
                await event_logger.log(
                    EventType.DATABASE_OPERATION,
                    "Database schema upgrade completed successfully",
                    Priority.HIGH,
                    metadata={"operation": "schema_ensure", "phase": "upgrade_complete"}
                )
                
            finally:
                # Always release the lock
                await event_logger.log(
                    EventType.DATABASE_OPERATION,
                    f"Releasing schema lock: {LOCK_ID}",
                    Priority.HIGH,
                    metadata={"operation": "schema_ensure", "phase": "lock_release", "lock_id": LOCK_ID}
                )
                
                stmt_unlock = select(
                    func.pg_advisory_unlock(bindparam("id", type_=BigInteger))
                ).params(id=LOCK_ID)
                await session.execute(stmt_unlock)
        
        duration = time.time() - start_time
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Schema initialization completed successfully in {duration:.2f}s",
            Priority.HIGH,
            metadata={
                "operation": "schema_ensure",
                "phase": "complete",
                "duration": duration,
                "success": True
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context="Database schema initialization",
            metadata={
                "operation": "schema_ensure",
                "duration": duration,
                "lock_id": LOCK_ID
            }
        )
        raise


async def commit_session(session: AsyncSession) -> None:
    """Explicitly commit the transaction on a connection."""
    
    start_time = time.time()
    await event_logger.log(
        EventType.DATABASE_OPERATION,
        "Committing database session",
        Priority.NORMAL,
        metadata={"operation": "session_commit"}
    )
    
    try:
        await session.commit()
        duration = time.time() - start_time
        await event_logger.log(
            EventType.DATABASE_OPERATION,
            f"Session committed successfully in {duration:.3f}s",
            Priority.NORMAL,
            metadata={
                "operation": "session_commit",
                "duration": duration,
                "success": True
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context="Database session commit",
            metadata={
                "operation": "session_commit",
                "duration": duration
            }
        )
        raise


__all__ = [
    "get_pg",
    "advisory_lock",
    "ensure_schema",
    "commit_session",
    "_maybe_await",
]
