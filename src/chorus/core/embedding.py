# src/chorus/core/embedding.py
"""Embedding utilities with optional naive fallback."""

from __future__ import annotations

import hashlib
from typing import Any

from chorus.config import config


async def embed_text(model: str, text: str) -> list[float]:
    """Return an embedding vector for ``text``.

    Parameters
    ----------
    model:
        Embedding model name.
    text:
        Text to embed.

    Returns
    -------
    list[float]
        Embedding vector of length 768.
    """

    import litellm

    selected = model
    api_base = config.embedding.api_base or config.llm.api_base
    api_key = config.embedding.api_key or config.llm.api_key
    if api_base and api_key:
        response: dict[str, Any] = await litellm.aembedding(
            model=selected,
            input=text,
            api_base=api_base,
            api_key=api_key,
        )
        return response["data"][0]["embedding"]

    # Fallback to naive hash-based embedding
    digest = hashlib.sha256(text.encode()).digest()
    vector = [(b - 128) / 128 for b in digest]
    dims = 768
    repeats = -(-dims // len(vector))  # ceil division
    expanded = (vector * repeats)[:dims]
    return expanded


__all__ = ["embed_text"]
