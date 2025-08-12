# src/chorus/core/logging.py
"""Logging helpers for Chorus."""

import json
import logging
import os
import sys
from pathlib import Path

from chorus.config import config

log_file = Path(config.system.log_file)

_LOGGING_INITIALIZED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach extras if present
        for key, value in record.__dict__.items():
            if key in (
                "args",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            ):
                continue
            # Avoid non-serializable objects
            try:
                json.dumps(value)
                data[key] = value
            except Exception:
                data[key] = str(value)
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def init_logging(
    level: str | None = None,
    format: str | None = None,
    include_trace: bool | None = None,
) -> None:
    """
    Initialize global logging configuration for Chorus.

    Environment variables:
      - CHORUS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL (default INFO)
      - CHORUS_LOG_FORMAT: plain|rich|json (default: auto rich if available, else plain)
      - CHORUS_LOG_INCLUDE_TRACE: bool (default False)
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    # Resolve settings
    env_level = os.getenv("CHORUS_LOG_LEVEL", "").upper() or "INFO"
    env_format = os.getenv("CHORUS_LOG_FORMAT", "")
    env_include_trace = os.getenv("CHORUS_LOG_INCLUDE_TRACE", "")

    resolved_level = (level or env_level or "INFO").upper()
    resolved_format = (format or env_format or "").lower()
    resolved_include_trace = (
        include_trace
        if include_trace is not None
        else _str_to_bool(env_include_trace, default=False)
    )

    # Map level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(resolved_level, logging.INFO)

    # Root logger cleanup
    root = logging.getLogger()
    root.setLevel(log_level)
    for h in list(root.handlers):
        root.removeHandler(h)

    # Determine formatter/handler
    use_rich = False
    if not resolved_format:
        # Auto: prefer rich if available
        try:
            import rich  # noqa: F401

            use_rich = True
        except Exception:
            use_rich = False
        resolved_format = "rich" if use_rich else "plain"
    else:
        use_rich = resolved_format == "rich"

    handler: logging.Handler
    if resolved_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(JsonFormatter())
    elif use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(
                level=log_level,
                rich_tracebacks=resolved_include_trace,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=True,
            )
            # Use a simple message-only format; RichHandler shows time/level nicely
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
        except Exception:
            # Fallback to plain
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)

    root.addHandler(handler)

    # Reduce noisy libraries if needed
    for noisy in ("uvicorn", "asyncio", "httpx"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))

    # Emit initial banner
    banner = logging.getLogger("chorus.start")
    from importlib import import_module

    version = "unknown"
    try:
        pkg = import_module("src.chorus.__init__".replace("/", "."))
    except Exception:
        try:
            pkg = import_module("chorus")
        except Exception:
            pkg = None
    if pkg is not None:
        version = getattr(pkg, "__version__", version)

    banner.info(
        "Initializing logging | version=%s level=%s format=%s include_trace=%s",
        version,
        resolved_level,
        resolved_format,
        str(resolved_include_trace),
    )

    _LOGGING_INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger with the provided name, or module path if None.
    """
    return logging.getLogger(name or "chorus")
