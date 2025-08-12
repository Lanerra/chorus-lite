# src/chorus/canon/queries.py
"""Low-level SQL utilities and embedding search functions."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from chorus.canon.db import get_pg
from chorus.config import config
from chorus.core.embedding import embed_text
from chorus.models import Chapter, CharacterProfile, WorldAnvil
from chorus.models.sqlalchemy_models import SceneSQL


def _validate_embedding_dim(embedding: Sequence[float]) -> None:
    """Ensure ``embedding`` matches the database vector dimension."""

    expected = SceneSQL.__table__.c.embedding.type.dim
    if len(list(embedding)) != expected:
        raise ValueError(
            f"Embedding dimension {len(list(embedding))} does not match database dimension {expected}"
        )


async def store_scene_text(scene_id: UUID | int, text: str) -> None:
    """Update a scene's text and embedding."""

    model = config.embedding.model
    embedding = await embed_text(model, text)
    _validate_embedding_dim(embedding)
    async with get_pg() as session:
        await session.execute(
            sa_text(
                "UPDATE scene SET text = :text, embedding = (:emb)::vector WHERE id = :sid"
            ),
            {"text": text, "emb": embedding, "sid": scene_id},
        )
        await session.commit()


async def store_scene_text_conn(
    session: AsyncSession, scene_id: UUID | int, text: str
) -> None:
    """Update ``scene_id`` text and embedding using existing ``session``."""

    model = config.embedding.model
    embedding = await embed_text(model, text)
    _validate_embedding_dim(embedding)
    await session.execute(
        sa_text(
            "UPDATE scene SET text = :text, embedding = (:emb)::vector WHERE id = :sid"
        ),
        {"text": text, "emb": embedding, "sid": scene_id},
    )


async def search_scene_text(text: str, *, limit: int = 5) -> list[UUID]:
    """Return scene IDs most similar to ``text``."""

    model = config.embedding.model
    embedding = await embed_text(model, text)
    _validate_embedding_dim(embedding)
    async with get_pg() as session:
        result = await session.execute(
            sa_text(
                "SELECT id FROM scene ORDER BY embedding <-> (:emb)::vector LIMIT :lim"
            ),
            {"emb": embedding, "lim": limit},
        )
        rows = result.scalars().all()
    return list(rows)


async def get_all_character_profiles(session: AsyncSession) -> list[CharacterProfile]:
    """Return minimal information about all characters."""

    result = await session.execute(
        sa_text("SELECT id, name FROM character_profile ORDER BY name")
    )
    rows = result.fetchall()
    return [CharacterProfile(id=r[0], name=r[1]) for r in rows]


async def get_all_world_anvils(session: AsyncSession) -> list[WorldAnvil]:
    """Return minimal information about all world elements."""

    result = await session.execute(
        sa_text("SELECT id, name FROM world_anvil ORDER BY name")
    )
    rows = result.fetchall()
    return [WorldAnvil(id=r[0], name=r[1]) for r in rows]


async def get_all_chapters(session: AsyncSession) -> list[Chapter]:
    """Return minimal information about all chapters."""

    result = await session.execute(
        sa_text(
            "SELECT id, title, description FROM chapter ORDER BY order_index, created_at"
        )
    )
    rows = result.fetchall()
    return [Chapter(id=r[0], title=r[1], description=r[2]) for r in rows]


__all__ = [
    "_validate_embedding_dim",
    "store_scene_text",
    "store_scene_text_conn",
    "search_scene_text",
    "get_all_character_profiles",
    "get_all_world_anvils",
    "get_all_chapters",
]
