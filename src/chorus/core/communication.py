# src/chorus/core/communication.py
"""Async Pub/Sub utilities for agent messaging."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """Message exchanged between agents."""

    sender: str = Field(..., min_length=1)
    recipient: str = Field(..., min_length=1)
    content: dict[str, Any] = Field(default_factory=dict)


_channels: defaultdict[str, asyncio.Queue[str]] = defaultdict(asyncio.Queue)


async def publish(message: AgentMessage) -> None:
    """Publish ``message`` to the recipient channel."""
    await _channels[message.recipient].put(message.model_dump_json())


async def subscribe(channel: str) -> AsyncGenerator[AgentMessage, None]:
    """Yield :class:`AgentMessage` objects from ``channel``."""
    queue = _channels[channel]
    try:
        while True:
            data = await queue.get()
            yield AgentMessage(**json.loads(data))
    finally:
        _channels.pop(channel, None)


async def close() -> None:
    """Clear all queued messages."""

    _channels.clear()


__all__ = ["AgentMessage", "publish", "subscribe", "close"]
