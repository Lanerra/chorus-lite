# src/chorus/models/base_model.py
"""Shared Pydantic base model with tolerant UUID handling."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChorusBaseModel(BaseModel):
    """Base model that converts invalid UUID strings to ``None``."""

    # Relax extra handling to ignore unexpected keys from LLMs instead of failing validation.
    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        validate_assignment=True,
    )

    @staticmethod
    def _coerce_uuids_rec(data: Any) -> None:
        """Recursively sanitize UUID fields within ``data``."""

        if not isinstance(data, dict):
            return

        for field_name, value in list(data.items()):
            if isinstance(value, dict):
                ChorusBaseModel._coerce_uuids_rec(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        ChorusBaseModel._coerce_uuids_rec(item)

            if isinstance(value, str) and "id" in field_name.lower():
                try:
                    UUID(value)
                except (ValueError, TypeError):
                    data[field_name] = None
            elif field_name == "dependencies" and isinstance(value, list):
                valid_dependencies: list[str | UUID] = []
                for item in value:
                    if isinstance(item, str):
                        try:
                            UUID(item)
                            valid_dependencies.append(item)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(item, UUID):
                        valid_dependencies.append(item)
                data[field_name] = valid_dependencies

    @model_validator(mode="before")
    @classmethod
    def _coerce_uuids(cls, data: Any) -> Any:
        cls._coerce_uuids_rec(data)
        if isinstance(data, dict):
            for field_name, field_type in cls.__annotations__.items():
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    value = data.get(field_name)
                    if isinstance(value, str):
                        for member in field_type:
                            if value.lower() in {
                                member.name.lower(),
                                str(member.value).lower(),
                            }:
                                data[field_name] = member
                                break
        return data


__all__ = ["ChorusBaseModel"]
