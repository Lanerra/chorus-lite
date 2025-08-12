# src/chorus/agents/base.py
"""Base class for Chorus agents."""

from __future__ import annotations

import os
import traceback
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession as PgSession
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from chorus.canon.db import _maybe_await
from chorus.canon.postgres import get_pg
from chorus.core.llm import call_llm, call_llm_structured
from chorus.core.logs import log_calls, log_message
from chorus.core.queue import dequeue, enqueue, mark_complete, move_to_dlq
from chorus.models import SceneStatus
from chorus.models.task import BaseTask

T = TypeVar("T", bound=BaseModel)


class Agent:
    """Base class for all Chorus agents, providing common utilities."""

    def __init__(self, *, model: str | None = None, default_model_env: str) -> None:
        """Initialize the agent with a model.

        The model can be passed directly or configured via an environment
        variable specified by `default_model_env`.

        Parameters
        ----------
        model:
            The LLM model name.
        default_model_env:
            The environment variable for the default model.

        Raises
        ------
        ValueError
            If no model is provided and the environment variable is not set.
        """
        self.model: str = cast(str, model or os.environ.get(default_model_env) or "")
        if not self.model:
            raise ValueError(f"Model not provided and {default_model_env} not set.")

    @log_calls
    async def call_llm(self, prompt: str) -> str:
        """Call the configured LLM with error logging."""

        try:
            return await call_llm(self.model, prompt)
        except Exception as exc:
            await self.log_message(f"LLM error: {exc}")
            raise

    @log_calls
    async def _call_llm(self, prompt: str) -> str:
        """Backward compatible wrapper for :meth:`call_llm`."""

        return await self.call_llm(prompt)

    @log_calls
    async def call_llm_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> T:
        """Call the LLM with structured output validation and error logging.

        Parameters
        ----------
        prompt:
            Prompt text to send to the LLM.
        response_model:
            Pydantic model describing expected structured output.
        temperature:
            Optional sampling temperature override. If None, global/default is used.
        max_retries:
            Maximum number of validation retry attempts.
        """
        try:
            # mypy: call_llm_structured expects float; only pass when not None
            kwargs: dict[str, Any] = {"max_retries": max_retries}
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            return await call_llm_structured(
                self.model,
                prompt,
                response_model,
                **kwargs,
            )
        except Exception as exc:
            await self.log_message(f"LLM error: {exc}")
            raise

    @log_calls
    async def _call_llm_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> T:
        """Backward compatible wrapper for :meth:`call_llm_structured`.

        Accepts optional temperature/max_retries to avoid breaking existing call sites.
        """
        return await self.call_llm_structured(
            prompt, response_model, temperature=temperature, max_retries=max_retries
        )

    @log_calls
    async def with_retries(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        retries: int = 3,
        wait_seconds: float = 0.1,
        **kwargs: Any,
    ) -> T:
        """Execute ``func`` with retry logic."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retries),
            wait=wait_fixed(wait_seconds),
            retry=retry_if_exception_type(Exception),
        ):
            with attempt:
                return await func(*args, **kwargs)

        raise RuntimeError("Unreachable")  # pragma: no cover - safety

    @log_calls
    async def enqueue_task(self, task: BaseTask, *, priority: int = 1) -> UUID:
        """Enqueue a task with optional priority."""
        return await enqueue(task, priority=priority)

    @log_calls
    async def dequeue_task(self) -> BaseTask | None:
        """Dequeue the highest priority ready task."""
        return await dequeue()

    @log_calls
    async def complete_task(self, task_id: UUID) -> None:
        """Mark a task as completed."""
        await mark_complete(task_id)

    @log_calls
    def get_db_connection(self) -> AbstractAsyncContextManager[PgSession]:
        """Get a PostgreSQL database connection."""
        return get_pg()

    @log_calls
    async def update_scene_status(
        self, scene_id: UUID, new_status: SceneStatus
    ) -> None:
        """Update the status of a scene in the database."""
        async with self.get_db_connection() as conn:
            await conn.execute(
                sa_text("UPDATE scene SET status = :status WHERE id = :sid"),
                {"status": new_status.value, "sid": scene_id},
            )
            await conn.commit()

    @log_calls
    async def get_scene_status(self, scene_id: UUID) -> SceneStatus:
        """Get the current status of a scene from the database."""
        async with self.get_db_connection() as conn:
            result = await conn.execute(
                sa_text("SELECT status FROM scene WHERE id = :sid"), {"sid": scene_id}
            )
            row = await _maybe_await(result.fetchone())
            if row is None:
                raise ValueError(f"Scene {scene_id} not found")
            return SceneStatus(row[0])

    @log_calls
    async def log_message(self, message: str) -> None:
        """Log a message with the agent's name."""
        log_message(f"{self.__class__.__name__}: {message}")

    @log_calls
    async def handle_task(
        self,
        task: BaseTask,
        handler: Callable[[BaseTask], Awaitable[None]],
        *,
        max_task_retries: int = 3,
        handler_retries: int = 3,
    ) -> bool:
        """Process ``task`` using ``handler`` with retry and DLQ handling."""

        try:
            await self.with_retries(handler, task, retries=handler_retries)
        except Exception as exc:
            await self.log_message(
                f"Task {task.id} failed with {exc}\n{traceback.format_exc()}"
            )
            task.attempts += 1
            if task.attempts > max_task_retries:
                await move_to_dlq(task)
                await self.log_message(f"Task {task.id} failed; moved to DLQ")
            else:
                await self.enqueue_task(task)
                await self.log_message(
                    f"Task {task.id} failed; requeued ({task.attempts}/{max_task_retries})"
                )
            return False

        if task.id is not None:
            await self.complete_task(task.id)
        await self.log_message(f"Task {task.id} completed")
        return True


__all__ = ["Agent"]
