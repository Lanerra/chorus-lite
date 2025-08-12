# src/chorus/core/cache.py
"""In-memory cache utilities used for temporary storage."""

from __future__ import annotations

from typing import TypeVar, cast

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_store: dict[str, str] = {}


async def set_model(key: str, value: BaseModel, *, ex: int | None = None) -> None:
    """Store ``value`` as JSON under ``key`` in the in-memory cache.

    Parameters
    ----------
    key:
        Cache key used to store ``value``.
    value:
        Pydantic model to cache.
    ex:
        Optional expiration time in seconds.
    """
    _store[key] = value.model_dump_json()


async def get_model(key: str, cls: type[T]) -> T | None:  # noqa: UP047
    """Return a ``cls`` instance stored at ``key`` if present."""
    data = _store.get(key)
    if data is None:
        return None
    return cast(T, cls.model_validate_json(data))


async def delete(key: str) -> None:
    """Remove ``key`` from the in-memory cache if it exists."""
    _store.pop(key, None)


async def close() -> None:
    """Clear all entries from the in-memory cache."""
    _store.clear()


__all__ = ["set_model", "get_model", "delete", "close"]
