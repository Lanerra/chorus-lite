# src/chorus/langgraph/errors.py
"""Custom exceptions for LangGraph nodes."""

from __future__ import annotations


class GraphRetry(Exception):
    """Signal that a node should be retried."""


__all__ = ["GraphRetry"]
