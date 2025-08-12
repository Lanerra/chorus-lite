# scripts/init_db.py
"""Run Alembic migrations to initialize the database."""

from __future__ import annotations

from pathlib import Path

from alembic import command as alembic_command  # type: ignore[attr-defined]
from alembic.config import Config

from chorus.core.logging import get_logger, init_logging

logger = get_logger(__name__)


async def init_db() -> None:
    """Apply Alembic migrations."""
    logger.info("Applying Alembic migrations to initialize the database")

    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    try:
        alembic_command.upgrade(cfg, "head")  # type: ignore[arg-type]
        logger.info("Alembic migrations applied successfully")
    except Exception as e:
        logger.exception("Failed to apply Alembic migrations: %s", e)
        raise


if __name__ == "__main__":  # pragma: no cover - CLI execution
    import asyncio

    init_logging()
    asyncio.run(init_db())
