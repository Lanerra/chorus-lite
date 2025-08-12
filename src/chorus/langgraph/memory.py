# src/chorus/langgraph/memory.py
"""Simple memory store utilities for LangGraph nodes."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import TypeAlias

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# Postgres-backed memory is deprecated; force in-memory only.
AsyncPostgresStore = None  # type: ignore

from chorus.config import config
from chorus.core.embedding import embed_text

Namespace: TypeAlias = tuple[str, ...]  # noqa: UP040


async def _embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Return embedding vectors for ``texts``."""

    model = config.embedding.model
    return await asyncio.gather(*(embed_text(model, t) for t in texts))


MEMORY_STORE = InMemoryStore(
    index={"dims": 768, "embed": _embed_texts, "fields": ["text"]}
)


@asynccontextmanager
async def get_memory_store() -> AsyncIterator[BaseStore]:
    """Yield the in-memory long-term memory store.

    Postgres-backed memory is deprecated and removed. Chorus uses ONLY the in-memory store.
    """
    yield MEMORY_STORE


async def store_text(namespace: Namespace, text: str) -> None:
    """Store ``text`` in long-term memory under ``namespace``."""

    key = str(uuid.uuid4())
    async with get_memory_store() as store:
        await store.aput(namespace, key, {"text": text})


async def search_text(namespace: Namespace, query: str, limit: int = 5) -> list[str]:
    """Return ``limit`` texts most relevant to ``query`` from ``namespace``.

    This function is resilient to missing backing tables when Postgres is configured:
    it will fall back to in-memory search and return an empty list if no memory exists yet.
    """

    async with get_memory_store() as store:
        results = await store.asearch(namespace, query=query, limit=limit)
    return [r.value.get("text", "") for r in results]


def summarize_text(text: str, *, limit: int = 50) -> str:
    """Return a short summary of ``text`` limited to ``limit`` words."""

    words = text.split()
    return " ".join(words[:limit])


__all__ = [
    "MEMORY_STORE",
    "get_memory_store",
    "store_text",
    "search_text",
    "summarize_text",
]
