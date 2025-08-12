# src/chorus/core/env.py
"""Environment configuration utilities."""

from __future__ import annotations

from dotenv import load_dotenv

from ..config import ChorusConfig, config


def load_env() -> None:
    """Load environment variables from a local ``.env`` file."""
    load_dotenv()

def get_config() -> ChorusConfig:
    """Get the global configuration instance."""
    return config

def get_settings() -> ChorusConfig:
    """Get the application settings."""
    return config



__all__ = ["load_env", "get_config", "get_settings"]
