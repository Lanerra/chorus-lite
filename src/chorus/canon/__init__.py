# src/chorus/canon/__init__.py
"""Database access helpers for the Canon."""

from .postgres import advisory_lock, get_pg

__all__ = ["get_pg", "advisory_lock"]
