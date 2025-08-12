# src/chorus/bootstrap.py
"""System bootstrap orchestrator and seed helpers.

This module now provides a unified async bootstrap that:
- Loads environment config
- Waits for PostgreSQL then applies migrations (ensure_schema)
- Warms the spaCy NER pipeline
- Exposes readiness and status inspection APIs

Seed helpers from previous implementation are preserved unchanged.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from chorus.core.env import load_env

try:
    # Prefer structured project logger if available
    from chorus.core.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging as _logging

    def get_logger(name: str | None = None):  # type: ignore
        return _logging.getLogger(name or "chorus")


from chorus.canon.db import ensure_schema, get_pg

# Removed NER import

# --------- Orchestrator state ---------

_IS_READY: bool = False


@dataclass
class _BootstrapStatus:
    started_at: float
    finished_at: float | None
    pg_ready: bool
    pg_migrated: bool
    ner_warmed: bool
    steps: list[dict[str, Any]]
    error: str | None


_STATUS: _BootstrapStatus = _BootstrapStatus(
    started_at=0.0,
    finished_at=None,
    pg_ready=False,
    pg_migrated=False,
    ner_warmed=False,
    steps=[],
    error=None,
)

logger = get_logger(__name__)


class BootstrapError(RuntimeError):
    """Bootstrap failed unexpectedly."""


class BootstrapTimeout(TimeoutError):
    """Bootstrap step exceeded timeout."""


def _set_readiness(flag: bool) -> None:
    """Set module-level readiness flag."""
    global _IS_READY
    _IS_READY = flag


def is_ready() -> bool:
    """Return True if bootstrap completed successfully."""
    return _IS_READY


def bootstrap_status() -> dict[str, object]:
    """Return a copy of current bootstrap status."""
    return asdict(_STATUS)


async def _wait_for_postgres(
    *, timeout: float, backoff_initial: float, backoff_factor: float, max_attempts: int
) -> None:
    """Wait for PostgreSQL by running SELECT 1 with exponential backoff."""
    attempt = 0
    delay = backoff_initial
    start = time.monotonic()
    while True:
        attempt += 1
        try:
            from sqlalchemy import text as sa_text

            async with get_pg() as session:
                await session.execute(sa_text("SELECT 1"))
            _STATUS.pg_ready = True
            logger.info("bootstrap.pg.ready", extra={"attempt": attempt})
            return
        except Exception as exc:
            elapsed = time.monotonic() - start
            _STATUS.steps.append(
                {"step": "pg_wait", "attempt": attempt, "error": str(exc)}
            )
            if elapsed > timeout or attempt >= max_attempts:
                raise BootstrapTimeout(
                    f"PostgreSQL wait timed out after {elapsed:.1f}s; last error: {exc}"
                ) from exc
            logger.warning(
                "bootstrap.pg.retry", extra={"attempt": attempt, "delay": delay}
            )
            await asyncio.sleep(delay)
            delay *= backoff_factor


async def _run_pg_migrations() -> None:
    """Apply Alembic migrations via ensure_schema()."""
    await ensure_schema()
    _STATUS.pg_migrated = True
    logger.info("bootstrap.pg.migrated")


def _warm_ner(warm: bool) -> None:
    """Preload spaCy NER pipeline for faster first request."""
    if not warm:
        return
    try:
        # Import and initialize NER pipeline if needed
        from chorus.core.ner import get_ner_pipeline  # type: ignore

        get_ner_pipeline()
        _STATUS.ner_warmed = True
        logger.info("bootstrap.ner.warmed")
    except Exception as exc:  # pragma: no cover - best-effort warm
        _STATUS.steps.append({"step": "ner_warm", "error": str(exc)})
        logger.warning("bootstrap.ner.warm_failed", extra={"error": str(exc)})


async def bootstrap_all(
    *,
    timeout_pg: float = 60.0,
    backoff_initial: float = 0.5,
    backoff_factor: float = 1.5,
    max_attempts: int = 20,
    warm_ner: bool = True,
) -> None:
    """Run full bootstrap sequence, fail-fast on irrecoverable errors.

    Steps:
      1) Load environment
      2) Wait for Postgres, then run migrations
      3) Warm NER
      4) Mark ready
    """
    if is_ready():
        logger.info("bootstrap.already_ready")
        return

    _STATUS.started_at = time.time()
    _STATUS.finished_at = None
    _STATUS.steps.clear()
    _STATUS.error = None
    _set_readiness(False)

    # 1) Load environment
    logger.info("bootstrap.env.load.start")
    load_env()
    logger.info("bootstrap.env.load.end")

    # Validate Postgres URL early
    pg_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRES_DSN")
        or ""
    )
    # Our config/db module already reads from config, but environment presence is required for startup clarity.
    if not pg_url:
        # Still allow chorus.config.config to provide a URL; try to import and read
        try:
            from chorus.config import config as _cfg  # type: ignore

            pg_url = getattr(_cfg.database, "postgres_url", "")  # type: ignore[attr-defined]
        except Exception:
            pg_url = ""
    if not pg_url:
        _STATUS.error = "Missing PostgreSQL connection URL (DATABASE_URL or config.database.postgres_url)"
        logger.error("bootstrap.pg.url_missing")
        raise BootstrapError(_STATUS.error)

    # 2) Wait for Postgres and run migrations
    logger.info("bootstrap.pg.wait.start")
    await _wait_for_postgres(
        timeout=timeout_pg,
        backoff_initial=backoff_initial,
        backoff_factor=backoff_factor,
        max_attempts=max_attempts,
    )
    logger.info("bootstrap.pg.wait.end")

    logger.info("bootstrap.pg.migrate.start")
    await _run_pg_migrations()
    logger.info("bootstrap.pg.migrate.end")

    # 3) Warm NER
    logger.info("bootstrap.ner.warm.start")
    _warm_ner(warm_ner)
    logger.info("bootstrap.ner.warm.end")

    # 4) Ready
    _set_readiness(True)
    _STATUS.finished_at = time.time()
    logger.info("bootstrap.ready", extra={"status": bootstrap_status()})


# --------- Seed helpers (unchanged) ---------

from chorus.agents.story_architect import StoryArchitect
from chorus.models import Concept


async def seed_concept(
    concept: Concept,
    *,
    story_architect_model: str,
) -> None:
    """Populate the databases with initial data for ``concept``."""
    # Create StoryArchitect instance to call methods
    architect = StoryArchitect(model=story_architect_model)
    world_entries = await architect.generate_world_anvil(concept)

    # TODO: Implement actual seeding logic with world_entries
    # For now, we'll just log that we generated entries
    logger.info("seed_concept.generated_entries", extra={"count": len(world_entries)})
