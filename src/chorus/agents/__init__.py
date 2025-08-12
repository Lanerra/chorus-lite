# src/chorus/agents/__init__.py
"""Agent implementations and utilities."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    # Base agent and utilities
    "Agent": "chorus.agents.base.Agent",
    # New consolidated agents (preferred)
    "StoryArchitect": "chorus.agents.story_architect.StoryArchitect",
    "SceneGenerator": "chorus.agents.scene_generator.SceneGenerator",
    "IntegrationManager": "chorus.agents.integration_manager.IntegrationManager",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """Load attributes lazily to avoid circular imports."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr = target.rsplit(".", 1)
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
