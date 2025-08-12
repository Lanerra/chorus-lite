# src/chorus/canon/db.py
"""Database session creation, migrations, and helpers."""

from __future__ import annotations

import inspect
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
from chorus.core.logs import log_message

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
    try:
        async with SessionLocal() as session:
            yield session
    except SQLAlchemyError as exc:  # pragma: no cover - connection errors
        log_message(f"PostgreSQL connection error: {exc}")
        raise


@asynccontextmanager
async def advisory_lock(session: AsyncSession, lock_id: int) -> AsyncIterator[None]:
    """Acquire a PostgreSQL advisory lock.

    Uses explicit bigint casts to avoid psycopg/SQLAlchemy binding as NUMERIC.
    """

    # Use typed bindparam (BIGINT) to satisfy pg_advisory_lock signature with psycopg3
    stmt_lock = select(func.pg_advisory_lock(bindparam("id", type_=BigInteger))).params(
        id=int(lock_id)
    )
    await session.execute(stmt_lock)
    try:
        yield
    finally:
        stmt_unlock = select(
            func.pg_advisory_unlock(bindparam("id", type_=BigInteger))
        ).params(id=int(lock_id))
        await session.execute(stmt_unlock)


async def ensure_schema() -> None:
    """Initialize extensions and apply migrations once using an advisory lock.

    Prevents infinite re-entry when multiple startup paths call ensure_schema()
    (e.g., Uvicorn reloaders, multiple nodes). Uses a session-level advisory
    lock and stamps baseline if DB is empty before upgrading to heads.
    """
    # Use a stable advisory lock id (signed 64-bit). Hash of 'chorus_schema_lock'.
    # Truncate to 63 bits to ensure it fits into BIGINT and retains sign safety.
    _RAW_LOCK_ID = 0x43686F7275735F534348454D41
    LOCK_ID = int(_RAW_LOCK_ID & 0x7FFF_FFFF_FFFF_FFFF)

    async with get_pg() as session:
        # Acquire advisory lock to serialize schema setup across processes
        stmt_lock = select(
            func.pg_advisory_lock(bindparam("id", type_=BigInteger))
        ).params(id=LOCK_ID)
        await session.execute(stmt_lock)
        try:
            # If any admin table exists, assume migrations have been applied sufficiently.
            result = await session.execute(
                sa_text("SELECT to_regclass('public.alembic_version')")
            )
            alembic_present = await _maybe_await(result.scalar())

            # If alembic_version is missing, stamp baseline then upgrade heads
            root = Path(__file__).resolve().parents[3]
            cfg = Config(str(root / "alembic.ini"))

            if alembic_present is None:
                # Stamp the initial baseline revision without running it repeatedly
                alembic_command.stamp(cfg, "20250805_baseline")  # type: ignore[attr-defined]

            # Now upgrade to latest heads exactly once
            alembic_command.upgrade(cfg, "heads")  # type: ignore[attr-defined]
        finally:
            # Always release the lock
            stmt_unlock = select(
                func.pg_advisory_unlock(bindparam("id", type_=BigInteger))
            ).params(id=LOCK_ID)
            await session.execute(stmt_unlock)


async def commit_session(session: AsyncSession) -> None:
    """Explicitly commit the transaction on a connection."""
    await session.commit()


__all__ = [
    "get_pg",
    "advisory_lock",
    "ensure_schema",
    "commit_session",
    "_maybe_await",
]
