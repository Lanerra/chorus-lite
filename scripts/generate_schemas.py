# scripts/generate_schemas.py
"""Generate JSON schemas for all Pydantic models."""

from __future__ import annotations

import json
from inspect import isclass
from pathlib import Path

import chorus.models as models
from chorus.models import ChorusBaseModel


def iter_models() -> list[type[ChorusBaseModel]]:
    """Return all public models that subclass :class:`ChorusBaseModel`."""
    result: list[type[ChorusBaseModel]] = []
    for name in getattr(models, "__all__", []):
        obj = getattr(models, name, None)
        if isclass(obj) and issubclass(obj, ChorusBaseModel):
            result.append(obj)
    return result


def main() -> None:  # pragma: no cover - script entry
    """Generate schemas in the ``docs/schemas`` directory."""
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "schemas"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in iter_models():
        schema = model.model_json_schema()
        path = out_dir / f"{model.__name__}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
