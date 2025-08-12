# src/chorus/core/output_utils.py
"""Utilities for writing debug logs and narrative outputs to disk.

This module centralizes safe file output for:
- Global LLM debug logging to src/chorus/debug/
- Narrative exports (scenes and chapters) to src/chorus/output/

Design goals:
- No external dependencies
- Async-friendly (uses asyncio.to_thread for blocking I/O)
- Safe concurrent writes (best-effort; last-write-wins is acceptable for debug/assembled outputs)
- UTF-8 text files; ensure directory creation
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path

# Base folders (as requested)
DEBUG_DIR = Path("src/chorus/debug")
OUTPUT_DIR = Path("src/chorus/output")

# Ensure top-level directories exist (lazy creation on write as well)
for p in (DEBUG_DIR, OUTPUT_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best effort; we will try again during writes
        pass


def _sanitize_name(name: str) -> str:
    """Sanitize a string for safe filename usage."""
    # Replace path separators and unsafe characters
    name = name.strip().replace(os.sep, "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _timestamp_ms() -> str:
    return str(int(time.time() * 1000))


async def _write_text(path: Path, content: str) -> None:
    """Async-friendly text write using UTF-8 encoding."""
    if content is None:
        content = ""
    # Ensure parent exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Ignore parent creation race
        pass

    def _sync_write() -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    await asyncio.to_thread(_sync_write)


async def write_debug_snapshot(
    *,
    base_slug: str,
    part: str,
    header: str | None,
    body: str,
) -> Path:
    """Write a debug snapshot file.

    Args:
        base_slug: A logical base name to correlate request/response (e.g., "story_architect_prompt_12345").
        part: A suffix to distinguish "prompt" vs "response" etc. (e.g., "prompt", "response").
        header: Optional header metadata to write at the top as commented lines.
        body: The main content to write.
    Returns:
        Path to the written file.
    """
    try:
        ts = _timestamp_ms()
        name = f"{_sanitize_name(base_slug)}_{_sanitize_name(part)}_{ts}.txt"
        path = DEBUG_DIR / name
        if header:
            content = f"--- {header.strip()} ---\n\n{body}"
        else:
            content = body
        await _write_text(path, content)
        return path
    except Exception:
        # Do not raise; debug logging must not break the pipeline
        return (
            DEBUG_DIR / f"{_sanitize_name(base_slug)}_{_sanitize_name(part)}_ERROR.txt"
        )


def scene_filename(chapter_number: int, scene_number: int) -> str:
    """Return filename for a scene: CH<chapter>-SCENE<scene>.txt"""
    return f"CH{int(chapter_number)}-SCENE{int(scene_number)}.txt"


def chapter_filename(chapter_number: int) -> str:
    """Return filename for a chapter: Chapter <chapter>.txt"""
    return f"Chapter {int(chapter_number)}.txt"


async def write_scene_output(
    *,
    chapter_number: int,
    scene_number: int,
    text: str,
) -> Path:
    """Write a scene to the output folder with the required naming."""
    try:
        filename = scene_filename(chapter_number, scene_number)
        path = OUTPUT_DIR / filename
        await _write_text(path, text or "")
        return path
    except Exception:
        return OUTPUT_DIR / "SCENE_WRITE_ERROR.txt"


async def write_chapter_output(
    *,
    chapter_number: int,
    text: str,
) -> Path:
    """Write a full chapter to the output folder with the required naming."""
    try:
        filename = chapter_filename(chapter_number)
        path = OUTPUT_DIR / filename
        await _write_text(path, text or "")
        return path
    except Exception:
        return OUTPUT_DIR / "CHAPTER_WRITE_ERROR.txt"
