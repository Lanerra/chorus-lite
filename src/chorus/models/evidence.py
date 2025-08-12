# src/chorus/models/evidence.py
"""Evidence span models for NER provenance and auditing."""

from __future__ import annotations

from pydantic import Field

from .base_model import ChorusBaseModel as BaseModel


class EvidenceSpan(BaseModel):
    """A reference to a span of text that supports an extraction decision.

    Attributes:
        start: Byte/character offset (inclusive) from the beginning of the source text.
        end: Byte/character offset (exclusive) from the beginning of the source text.
        quote: The exact substring captured for human verification.
        source_id: Optional identifier of the source (e.g., scene UUID).
    """

    start: int = Field(..., ge=0, description="Inclusive start offset")
    end: int = Field(..., ge=0, description="Exclusive end offset")
    quote: str = Field("", description="Extracted substring for verification")
    source_id: str | None = Field(
        default=None, description="Optional logical source identifier"
    )


__all__ = ["EvidenceSpan"]
