# src/chorus/core/llm.py
"""Lightweight wrapper around LiteLLM for async LLM calls with structured output support."""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from typing import Any, TypeVar, get_origin

import dirtyjson
from pydantic import BaseModel, ValidationError

from chorus.config import config
from chorus.core.logs import get_event_logger, EventType, Priority
from chorus.core.output_utils import write_debug_snapshot

# Initialize EventLogger for LLM operations
event_logger = get_event_logger()

T = TypeVar("T", bound=BaseModel)


def _split_to_list(value: str) -> list[str]:
    """Heuristically split a long string into a list of trimmed non-empty items."""
    # Try common delimiters and bullet patterns
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    # Normalize bullets to newlines
    text = re.sub(r"[•\u2022]", "\n", text)
    # Split by numbered bullets or semicolons/newlines
    parts = re.split(r"(?:\n|\\n|;|^\s*\d+\.\s+|- )+", text)
    items = [p.strip(" -\t\r\n") for p in parts if p and p.strip(" -\t\r\n")]
    if not items:
        return [text]
    return items


def _normalize_character_profiles(json_data: Any) -> Any:
    """
    Normalize common variants of CharacterProfileList LLM output into the expected schema:
    - Accept dict keyed by character name -> convert to list and inject name if missing.
    - Validate/normalize slug-like 'name' field (lowercase, hyphenated).
    """
    if json_data is None:
        return json_data

    # If top-level is a single-key dict wrapping a list, unwrap
    if (
        isinstance(json_data, dict)
        and len(json_data) == 1
        and isinstance(next(iter(json_data.values())), list)
    ):
        json_data = next(iter(json_data.values()))

    # If top-level is dict keyed by names, convert to list
    if isinstance(json_data, dict):
        items = []
        for k, v in json_data.items():
            if isinstance(v, dict):
                item = dict(v)
                if "name" not in item or not item.get("name"):
                    item["name"] = str(k)
                items.append(item)
        json_data = items

    # If single dict provided, wrap in list
    if isinstance(json_data, dict):
        json_data = [json_data]

    if isinstance(json_data, list):
        normalized = []
        for item in json_data:
            if not isinstance(item, dict):
                continue
            obj = dict(item)
            # Ensure name exists
            name = obj.get("name")
            if not name and "full_name" in obj:
                name = obj.get("full_name")
            if not name:
                # try keys like id, slug
                name = obj.get("slug") or obj.get("id")
            if name:
                slug = re.sub(r"[^a-z0-9-]+", "-", str(name).lower()).strip("-")
                slug = re.sub(r"-{2,}", "-", slug)
                obj["name"] = slug

            # Fields to coerce into list[str]
            for key in ("beliefs", "desires", "intentions", "motivations", "aliases"):
                val = obj.get(key)
                if isinstance(val, str):
                    obj[key] = _split_to_list(val)
                elif isinstance(val, list):
                    # Trim all and drop empties, ensure strings
                    coerced = []
                    for x in val:
                        if isinstance(x, str):
                            sx = x.strip()
                            if sx:
                                coerced.append(sx)
                        else:
                            sx = str(x).strip()
                            if sx:
                                coerced.append(sx)
                    obj[key] = coerced
                elif val is None:
                    obj[key] = []
                else:
                    # Unknown type -> stringify
                    obj[key] = _split_to_list(str(val))

            # Normalize any date fields if present to ISO strings (pydantic can parse dates)
            for key in ("birth_date", "death_date"):
                d = obj.get(key)
                if isinstance(d, datetime | date):
                    obj[key] = d.isoformat()
                elif isinstance(d, str):
                    obj[key] = d.strip()

            normalized.append(obj)
        json_data = normalized

    return json_data


def _normalize_relationship_triples(json_data: Any) -> Any:
    """
    Normalize common variants of RelationshipTripleList LLM output into the expected schema:
    - Accept dicts or lists under a single key wrapping the list.
    - Accept objects using alias keys like subject/from/from_id -> source,
      object/to/target_id -> target, type/edge/relationship -> relation.
    - Strip Markdown fences or prose wrappers are handled upstream.
    """
    if json_data is None:
        return json_data

    # Unwrap {"relationships":[...]} or similar single-key wrappers
    if isinstance(json_data, dict):
        # If dict is actually an error container, leave it to upstream handling
        if any(k in json_data for k in ("error", "errors", "detail")):
            return json_data
        if len(json_data) == 1 and isinstance(next(iter(json_data.values())), list):
            json_data = next(iter(json_data.values()))
        else:
            # Sometimes models return a dict representing a single triple
            # Convert it into a single-item list for uniform handling
            json_data = [json_data]

    # If single object, wrap into a list
    if isinstance(json_data, dict):
        json_data = [json_data]

    if not isinstance(json_data, list):
        return json_data

    normalized: list[dict[str, Any]] = []
    for item in json_data:
        if not isinstance(item, dict):
            # If item is a string like "a->b:rel", try to parse rudimentarily
            if isinstance(item, str):
                txt = item.strip()
                m = re.match(
                    r"^\s*([^\-:>]+)\s*[-:>]+\s*([^\-:>]+)\s*[:\-]\s*(.+)$", txt
                )
                if m:
                    src, tgt, rel = m.groups()
                    normalized.append(
                        {
                            "source": str(src).strip(),
                            "target": str(tgt).strip(),
                            "relation": str(rel).strip(),
                        }
                    )
                continue
            continue

        # Lowercase keys for easier alias mapping (preserve original values)
        keys_lower = {k.lower(): k for k in item.keys()}

        def pick(keys: list[str]) -> Any:
            for k in keys:
                lk = k.lower()
                if lk in keys_lower:
                    return item[keys_lower[lk]]
            return None

        source = pick(["source", "from", "from_id", "subject", "start", "a"])
        target = pick(["target", "to", "to_id", "object", "end", "b"])
        relation = pick(["relation", "relationship", "type", "edge", "rel"])

        # Fallback if nested under 'data' or similar
        if source is None or target is None or relation is None:
            data = pick(["data", "triple", "relationship"])
            if isinstance(data, dict):
                item2 = data
                keys_lower2 = {k.lower(): k for k in item2.keys()}

                def pick2(keys: list[str]) -> Any:
                    for k in keys:
                        lk = k.lower()
                        if lk in keys_lower2:
                            return item2[keys_lower2[lk]]
                    return None

                source = source or pick2(
                    ["source", "from", "from_id", "subject", "start", "a"]
                )
                target = target or pick2(
                    ["target", "to", "to_id", "object", "end", "b"]
                )
                relation = relation or pick2(
                    ["relation", "relationship", "type", "edge", "rel"]
                )

        # Coerce to trimmed strings if present
        def to_str(x: Any) -> str | None:
            if x is None:
                return None
            s = str(x).strip()
            return s if s else None

        src_s = to_str(source)
        tgt_s = to_str(target)
        rel_s = to_str(relation)

        # Only include fully-formed triples
        if src_s and tgt_s and rel_s:
            normalized.append({"source": src_s, "target": tgt_s, "relation": rel_s})

    return normalized if normalized else json_data


# --- New normalization helpers and retryable error for structured pipeline hardening ---


class RetryableValidationError(Exception):
    """Signal a retryable structural validation problem with guidance for a stricter retry.

    Attributes:
        reason: Short description of the structural issue.
        expected_root: Expected root type ('array'|'object').
        stricter_suffix: Instructional suffix to append on retry to tighten format.
        example_json: Minimal example JSON to include in retry context.
    """

    def __init__(
        self,
        reason: str,
        expected_root: str,
        stricter_suffix: str,
        example_json: str,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.expected_root = expected_root
        self.stricter_suffix = stricter_suffix
        self.example_json = example_json


def _normalize_entity_list(json_data: Any) -> Any:
    """Normalize EntityList path.

    Heuristics:
    - Enforce array root: if a single dict that looks like an entity, wrap once.
    - If category-keyed dict (e.g., {'characters': [...]}), flatten into a single array,
      inferring type buckets from keys. Prefer schema adherence; if ambiguous, raise RetryableValidationError.
    """
    if json_data is None:
        return json_data

    # If already a list, accept
    if isinstance(json_data, list):
        return json_data

    # Single entity object -> wrap as array
    if isinstance(json_data, dict):
        # Category-keyed form?
        lower_keys = {str(k).lower(): k for k in json_data.keys()}
        bucket_map = {
            "characters": "CHARACTER",
            "character": "CHARACTER",
            "people": "CHARACTER",
            "persons": "CHARACTER",
            "locations": "LOCATION",
            "location": "LOCATION",
            "places": "LOCATION",
            "items": "ITEM",
            "item": "ITEM",
            "objects": "ITEM",
            "artifacts": "ITEM",
            "concepts": "CONCEPT",
            "concept": "CONCEPT",
            "ideas": "CONCEPT",
            "themes": "CONCEPT",
        }
        any_bucket = any(k in lower_keys for k in bucket_map.keys())
        if any_bucket:
            flat: list[dict[str, Any]] = []
            for lk, orig in lower_keys.items():
                etype = bucket_map.get(lk)
                if not etype:
                    continue
                v = json_data[orig]
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            name = it.get("name") or it.get("topic")
                            desc = it.get("description") or it.get("summary")
                            if name:
                                flat.append(
                                    {
                                        "name": str(name),
                                        "type": etype,
                                        "description": desc,
                                    }
                                )
                        elif isinstance(it, str):
                            flat.append({"name": it, "type": etype})
                elif isinstance(v, dict):
                    name = v.get("name") or v.get("topic")
                    desc = v.get("description") or v.get("summary")
                    if name:
                        flat.append(
                            {"name": str(name), "type": etype, "description": desc}
                        )
                elif isinstance(v, str):
                    flat.append({"name": v, "type": etype})
            if flat:
                return flat
            # Bucketed but we couldn't flatten -> prefer strict retry
            raise RetryableValidationError(
                reason="Category-keyed dict could not be flattened into Entity array",
                expected_root="array",
                stricter_suffix=(
                    "Return ONLY a JSON array of objects; do not wrap in an object. "
                    "Each object must include name (string) and type (enum: CHARACTER|LOCATION|ITEM|CONCEPT). "
                    "description is optional. No other keys."
                ),
                example_json='[{"name":"Kael","type":"CHARACTER","description":"A rogue."}]',
            )

        # Looks like a single entity?
        looks_like_entity = any(k in json_data for k in ("name", "type", "description"))
        if looks_like_entity:
            return [json_data]

    # If we reached here, wrong root
    raise RetryableValidationError(
        reason=f"Expected top-level JSON array for EntityList, got {type(json_data).__name__}",
        expected_root="array",
        stricter_suffix=(
            "Return ONLY a JSON array of objects; do not wrap in an object. "
            "Each object must include name (string) and type (enum). description is optional. No other keys."
        ),
        example_json='[{"name":"Kael","type":"CHARACTER","description":"A rogue."}]',
    )


def _normalize_graph_extraction(json_data: Any) -> Any:
    """Normalize GraphExtraction to enforce minimal contract and filter noise.

    Policy:
    - Enforce exact top-level keys: characters, locations, items, events, organizations, concepts, relationships.
      Drop any unknown keys before validation (pydantic extra='forbid' will also reject).
    - Coerce entity arrays to strings via existing validators; additionally filter generic/common-noun entries
      using a stoplist.
    - Normalize relationship triples to only include 'source','relation','target'. Map common relation aliases
      to a canonical set. Drop triples whose endpoints are not present in any entity arrays.
    """
    if json_data is None:
        return json_data

    allowed_keys = {
        "characters",
        "locations",
        "items",
        "events",
        "organizations",
        "concepts",
        "relationships",
    }

    # Drop extraneous top-level keys if dict
    if isinstance(json_data, dict):
        json_data = {k: v for k, v in json_data.items() if str(k) in allowed_keys}
    else:
        # Not an object -> retry with stricter contract
        raise RetryableValidationError(
            reason=f"Expected GraphExtraction to be a JSON object, got {type(json_data).__name__}",
            expected_root="object",
            stricter_suffix=(
                "Return a single JSON object with ONLY these top-level keys: "
                "characters, locations, items, events, organizations, concepts, relationships. "
                "No additional keys. Each list contains ONLY strings (entity names). "
                "relationships is an array of objects each with EXACTLY: source, relation, target. "
                "Do not include evidence. Field names and types must match exactly."
            ),
            example_json=(
                '{"characters":[],"locations":[],"items":[],"events":[],"organizations":[],'
                '"concepts":[],"relationships":[{"source":"alice","relation":"FRIEND_OF","target":"bob"}]}'
            ),
        )

    # Ensure all expected keys exist
    for k in list(allowed_keys):
        json_data.setdefault(k, [])

    # Filter entities using a small stoplist of generic/common terms
    stoplist = {
        "thing",
        "object",
        "item",
        "place",
        "city",
        "server room",
        "terminal",
        "system",
        "room",
        "building",
        "hall",
        "street",
        "man",
        "woman",
        "person",
        "people",
    }

    def _coerce_entity_list(val: Any) -> list[str]:
        names: list[str] = []
        if isinstance(val, list):
            for it in val:
                if isinstance(it, str):
                    names.append(it)
                elif isinstance(it, dict):
                    nm = it.get("name")
                    if isinstance(nm, str):
                        names.append(nm)
        elif isinstance(val, dict):
            nm = val.get("name")
            if isinstance(nm, str):
                names.append(nm)
        elif isinstance(val, str):
            names.append(val)
        return [
            n for n in (s.strip() for s in names) if n and n.lower() not in stoplist
        ]

    for key in (
        "characters",
        "locations",
        "items",
        "events",
        "organizations",
        "concepts",
    ):
        json_data[key] = _coerce_entity_list(json_data.get(key, []))

    # Normalize relationship triples
    canon_relations = {
        "LEADS_TO",
        "HAPPENS_AFTER",
        "CAUSES",
        "USES",
        "OWNS",
        "LOCATED_IN",
        "MEMBER_OF",
        "FRIEND_OF",
        "HAPPENS_BEFORE",
        "BELONGS_TO",
        "LIVES_IN",
        "OCCURS_IN",
        "TAKES_PLACE_IN",
        "RELATED_TO",
        "KNOWS",
        "ALLIES_WITH",
        "OPPOSES",
        "LOVES",
        "HATES",
        "SERVES",
        "APPEARS_IN",
        "FOLLOWS",
        "FORESHADOWS",
        "SYMBOLIZES",
        "REPRESENTS",
        "REVEALS",
        "EXPLAINS",
        "CONTAINS",
    }
    alias_map = {
        "HAS_AUTHORIZATION": "AUTHORIZED_FOR",
        "AUTHORIZED_FOR": "AUTHORIZED_FOR",
        "DISPLAYS": "SHOWS",
        "SHOWS": "SHOWS",
        "BELONGS-TO": "BELONGS_TO",
        "OCCURS-IN": "OCCURS_IN",
        "TAKES-PLACE-IN": "TAKES_PLACE_IN",
        "HAPPENS-AFTER": "HAPPENS_AFTER",
        "HAPPENS-BEFORE": "HAPPENS_BEFORE",
        "LOCATED-IN": "LOCATED_IN",
        "MEMBER-OF": "MEMBER_OF",
        "FRIEND-OF": "FRIEND_OF",
        "LEADS-TO": "LEADS_TO",
    }

    # Build entity presence set
    entity_set = set(
        n.strip()
        for key in (
            "characters",
            "locations",
            "items",
            "events",
            "organizations",
            "concepts",
        )
        for n in json_data.get(key, [])
        if isinstance(n, str) and n.strip()
    )

    def _clean_triple(t: Any) -> dict[str, str] | None:
        if not isinstance(t, dict):
            return None
        src = str(t.get("source") or t.get("from") or t.get("subject") or "").strip()
        tgt = str(t.get("target") or t.get("to") or t.get("object") or "").strip()
        rel = str(
            t.get("relation") or t.get("relationship") or t.get("type") or ""
        ).strip()
        if not (src and tgt and rel):
            return None
        # Canonicalize relation
        rel_up = re.sub(r"[^A-Z_]+", "_", rel.upper()).strip("_")
        rel_up = alias_map.get(rel_up, rel_up)
        if rel_up not in canon_relations:
            # If no safe mapping, drop the triple
            return None
        # Validate endpoints exist in any entity array
        if src not in entity_set or tgt not in entity_set:
            return None
        return {"source": src, "relation": rel_up, "target": tgt}

    rels_raw = json_data.get("relationships", [])
    cleaned: list[dict[str, str]] = []
    if isinstance(rels_raw, list):
        for t in rels_raw:
            ct = _clean_triple(t)
            if ct:
                cleaned.append(ct)
    elif isinstance(rels_raw, dict):
        ct = _clean_triple(rels_raw)
        if ct:
            cleaned.append(ct)
    json_data["relationships"] = cleaned
    return json_data


def _maybe_normalize_for_model(response_model: type[BaseModel], json_data: Any) -> Any:
    """Dispatch normalization based on target response_model name to increase robustness."""
    model_name = getattr(response_model, "__name__", "")
    if model_name == "CharacterProfileList":
        return _normalize_character_profiles(json_data)
    if model_name == "RelationshipTripleList":
        return _normalize_relationship_triples(json_data)
    if model_name == "EntityList":
        return _normalize_entity_list(json_data)
    if model_name == "GraphExtraction":
        return _normalize_graph_extraction(json_data)
    return json_data


def _get_global_temperature() -> float:
    """Return global temperature from env/config with default 0.7."""
    # Prefer explicit environment variable if present
    import os

    default = 0.7
    val = os.getenv("TEMPERATURE")
    try:
        if val is not None:
            t = float(val)
        else:
            # Fall back to config if available; otherwise default
            t = getattr(getattr(config, "llm", object()), "temperature", default)  # type: ignore[attr-defined]
            if t is None:
                t = default
    except Exception:
        t = default
    # Clamp to a reasonable range used by common providers
    if t < 0.0:
        t = 0.0
    if t > 2.0:
        t = 2.0
    return t


async def call_llm(model: str, prompt: str) -> str:
    """Call the configured LLM and return the generated text.

    Parameters
    ----------
    model:
        Name of the model to query.
    prompt:
        User prompt passed directly to the model.

    Returns
    -------
    str
        The LLM's response content.

    Raises
    ------
    RuntimeError
        If required environment variables are missing.
    """
    
    start_time = time.time()
    temperature = _get_global_temperature()
    prompt_length = len(prompt) if prompt else 0
    
    await event_logger.log(
        EventType.LLM_REQUEST,
        f"Starting LLM call to {model}",
        Priority.NORMAL,
        metadata={
            "operation": "call_llm",
            "model": model,
            "prompt_length": prompt_length,
            "temperature": temperature,
            "structured": False
        }
    )

    import litellm

    api_base = config.llm.api_base
    api_key = config.llm.api_key
    if not api_base or not api_key:
        await event_logger.log_error_handling_start(
            error_type="RuntimeError",
            error_msg="OPENAI_API_BASE and OPENAI_API_KEY must be set",
            context=f"LLM call to {model}",
            metadata={"operation": "call_llm", "model": model, "error_category": "configuration"}
        )
        raise RuntimeError("OPENAI_API_BASE and OPENAI_API_KEY must be set")

    # Debug: write prompt snapshot
    try:
        header = f"model={model}"
        await write_debug_snapshot(
            base_slug=f"llm_{model}",
            part="prompt",
            header=header,
            body=prompt or "",
        )
    except Exception:
        # Never fail the core path due to debug logging
        pass

    try:
        await event_logger.log(
            EventType.LLM_REQUEST,
            f"Sending request to {model}",
            Priority.NORMAL,
            metadata={
                "operation": "call_llm",
                "model": model,
                "api_base": api_base,
                "phase": "request_send"
            }
        )
        
        response: dict[str, Any] = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
        )
        content = response["choices"][0]["message"]["content"]
        
        duration = time.time() - start_time
        response_length = len(content) if content else 0
        
        await event_logger.log(
            EventType.LLM_REQUEST,
            f"Successfully received response from {model}",
            Priority.NORMAL,
            metadata={
                "operation": "call_llm",
                "model": model,
                "duration": duration,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "temperature": temperature,
                "success": True
            }
        )

        # Debug: write response snapshot
        try:
            header = f"model={model}"
            await write_debug_snapshot(
                base_slug=f"llm_{model}",
                part="response",
                header=header,
                body=content or "",
            )
        except Exception:
            pass

        return content
        
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"LLM call to {model}",
            metadata={
                "operation": "call_llm",
                "model": model,
                "duration": duration,
                "prompt_length": prompt_length,
                "temperature": temperature
            }
        )
        raise


async def call_llm_structured[T: BaseModel](  # type: ignore[valid-type]
    model: str,
    prompt: str,
    response_model: type[T],
    max_retries: int = 3,
    temperature: float = None,  # type: ignore[assignment]
) -> T:
    """Call the LLM with structured output validation and retry logic.

    Uses Pydantic model's JSON schema to constrain LLM output format.
    Automatically retries on validation failures with error feedback.

    Parameters
    ----------
    model:
        Name of the model to query.
    prompt:
        User prompt passed to the model.
    response_model:
        Pydantic model class that defines the expected output structure.
    max_retries:
        Maximum number of retry attempts on validation failures.
    temperature:
        Sampling temperature for LLM generation.

    Returns
    -------
    T
        Validated instance of the response_model.

    Raises
    ------
    RuntimeError
        If required environment variables are missing.
    ValidationError
        If validation fails after all retry attempts.
    """
    
    start_time = time.time()
    prompt_length = len(prompt) if prompt else 0
    model_name = getattr(response_model, '__name__', str(response_model))
    
    await event_logger.log(
        EventType.LLM_REQUEST,
        f"Starting structured LLM call to {model} for {model_name}",
        Priority.NORMAL,
        metadata={
            "operation": "call_llm_structured",
            "model": model,
            "response_model": model_name,
            "prompt_length": prompt_length,
            "max_retries": max_retries,
            "structured": True
        }
    )
    
    import litellm

    api_base = config.llm.api_base
    api_key = config.llm.api_key
    if not api_base or not api_key:
        await event_logger.log_error_handling_start(
            error_type="RuntimeError",
            error_msg="OPENAI_API_BASE and OPENAI_API_KEY must be set",
            context=f"Structured LLM call to {model} for {model_name}",
            metadata={
                "operation": "call_llm_structured",
                "model": model,
                "response_model": model_name,
                "error_category": "configuration"
            }
        )
        raise RuntimeError("OPENAI_API_BASE and OPENAI_API_KEY must be set")

    # Generate JSON schema from the Pydantic model
    schema = response_model.model_json_schema()
    last_error: ValidationError | Exception | None
    response: dict[str, Any]

    # Resolve effective temperature (global default if not explicitly provided)
    effective_temperature = (
        _get_global_temperature() if temperature is None else temperature
    )
    
    await event_logger.log(
        EventType.LLM_REQUEST,
        f"Generated JSON schema for {model_name}",
        Priority.NORMAL,
        metadata={
            "operation": "call_llm_structured",
            "model": model,
            "response_model": model_name,
            "temperature": effective_temperature,
            "schema_type": schema.get("type", "unknown"),
            "phase": "schema_generation"
        }
    )

    # Debug: write prompt snapshot (structured)
    try:
        header = f"model={model}; structured_response_model={getattr(response_model, '__name__', str(response_model))}; temperature={effective_temperature}"
        await write_debug_snapshot(
            base_slug=f"llm_structured_{model}",
            part="prompt",
            header=header,
            body=prompt or "",
        )
    except Exception:
        pass

    # For string schemas, enforce JSON-only by wrapping expected output into {"value": string}
    if schema.get("type") == "string":
        # Force a JSON object with a single key "value" and low temperature
        effective_temperature = (
            0.1
            if effective_temperature is None or effective_temperature > 0.2
            else effective_temperature
        )
        wrapped_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        # Strong instruction to return ONLY JSON
        system_instr = "Return ONLY valid JSON. No markdown, code fences, or prose."
        structured_prompt = f'{system_instr}\nRespond with a JSON object: {{"value": <string>}}.\n\nTask:\n{prompt}'
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": structured_prompt}],
                    api_base=api_base,
                    api_key=api_key,
                    temperature=effective_temperature,
                    response_format={"type": "json_schema", "schema": wrapped_schema},
                )
                content = response["choices"][0]["message"]["content"].strip()

                # Debug: write response snapshot for simple string schema branch
                try:
                    header = f"model={model}; structured=string; temperature={effective_temperature}"
                    await write_debug_snapshot(
                        base_slug=f"llm_structured_{model}",
                        part="response",
                        header=header,
                        body=content or "",
                    )
                except Exception:
                    pass
                # Enforce JSON-only decode
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n?", "", content)
                    content = re.sub(r"```$", "", content).strip()
                try:
                    obj = json.loads(content)
                except json.JSONDecodeError:
                    # Attempt to salvage JSON object
                    match = re.search(r"({.*})", content, re.DOTALL)
                    if not match:
                        raise
                    obj = dirtyjson.loads(match.group(1))
                if (
                    not isinstance(obj, dict)
                    or "value" not in obj
                    or not isinstance(obj["value"], str)
                ):
                    raise ValidationError.from_exception_data(
                        "ValidationError",
                        [
                            {
                                "type": "string_type",
                                "loc": ("root",),
                                "input": obj,
                            }
                        ],
                    )
                return response_model.model_validate(obj["value"])
            except ValidationError as e:
                last_error = e
                if attempt < max_retries:
                    structured_prompt = (
                        f"{prompt}\nPrevious attempt failed with error: {e}"
                    )
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error in structured LLM call")

    # Enforce JSON-only responses for non-string schemas with explicit instruction
    system_instr = "Return ONLY valid JSON. No markdown, code fences, or prose."
    structured_prompt = f"{system_instr}\n{prompt}"
    # Force low temperature for structured tasks
    if effective_temperature is None or effective_temperature > 0.2:
        effective_temperature = 0.1

    last_error = None
    stricter_suffix_used = None
    example_for_retry = None

    for attempt in range(max_retries + 1):
        attempt_start = time.time()
        await event_logger.log(
            EventType.LLM_REQUEST,
            f"Structured LLM attempt {attempt + 1}/{max_retries + 1} for {model_name}",
            Priority.NORMAL,
            metadata={
                "operation": "call_llm_structured",
                "model": model,
                "response_model": model_name,
                "attempt": attempt + 1,
                "max_attempts": max_retries + 1,
                "phase": "attempt_start"
            }
        )
        
        try:
            await event_logger.log(
                EventType.LLM_REQUEST,
                f"Sending structured request to {model} (attempt {attempt + 1})",
                Priority.NORMAL,
                metadata={
                    "operation": "call_llm_structured",
                    "model": model,
                    "response_model": model_name,
                    "attempt": attempt + 1,
                    "temperature": effective_temperature,
                    "phase": "request_send"
                }
            )
            
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": structured_prompt}],
                api_base=api_base,
                api_key=api_key,
                temperature=effective_temperature,
                response_format={"type": "json_schema", "schema": schema},
            )

            content = response["choices"][0]["message"]["content"].strip()
            content_length = len(content)
            request_time = time.time() - attempt_start
            
            await event_logger.log(
                EventType.LLM_REQUEST,
                f"Received response from {model} (attempt {attempt + 1})",
                Priority.NORMAL,
                metadata={
                    "operation": "call_llm_structured",
                    "model": model,
                    "response_model": model_name,
                    "attempt": attempt + 1,
                    "request_duration": request_time,
                    "response_length": content_length,
                    "phase": "response_received"
                }
            )

            # Debug: write response snapshot for JSON schema branch
            try:
                header = f"model={model}; structured_json_schema=true; temperature={effective_temperature}"
                await write_debug_snapshot(
                    base_slug=f"llm_structured_{model}",
                    part="response",
                    header=header,
                    body=content or "",
                )
            except Exception:
                pass

            # Remove Markdown code fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"```$", "", content).strip()

            # Parse and validate JSON response
            await event_logger.log(
                EventType.LLM_REQUEST,
                f"Parsing JSON response from {model} (attempt {attempt + 1})",
                Priority.NORMAL,
                metadata={
                    "operation": "call_llm_structured",
                    "model": model,
                    "response_model": model_name,
                    "attempt": attempt + 1,
                    "phase": "json_parsing"
                }
            )
            
            try:
                if (
                    response_model.__name__ == "CharacterProfileList"
                    and '"character"' in content
                    and not content.strip().startswith("[")
                ):
                    await event_logger.log(
                        EventType.LLM_REQUEST,
                        f"Applying CharacterProfileList parsing logic",
                        Priority.NORMAL,
                        metadata={
                            "operation": "call_llm_structured",
                            "model": model,
                            "response_model": model_name,
                            "attempt": attempt + 1,
                            "phase": "special_parsing"
                        }
                    )
                    matches = re.findall(
                        r"\"character\"\s*:\s*({.*?})(?=,\s*\"character\"\s*:|\s*}$)",
                        content,
                        re.DOTALL,
                    )
                    if matches:
                        json_data = json.loads("[" + ",".join(matches) + "]")
                    else:
                        json_data = json.loads(content)
                else:
                    json_data = json.loads(content)
                    
                await event_logger.log(
                    EventType.LLM_REQUEST,
                    f"Successfully parsed JSON response (attempt {attempt + 1})",
                    Priority.NORMAL,
                    metadata={
                        "operation": "call_llm_structured",
                        "model": model,
                        "response_model": model_name,
                        "attempt": attempt + 1,
                        "parsed_type": type(json_data).__name__,
                        "phase": "json_parsed"
                    }
                )
                    
            except json.JSONDecodeError as e:
                await event_logger.log(
                    EventType.LLM_REQUEST,
                    f"JSON parsing failed, attempting recovery (attempt {attempt + 1})",
                    Priority.NORMAL,
                    metadata={
                        "operation": "call_llm_structured",
                        "model": model,
                        "response_model": model_name,
                        "attempt": attempt + 1,
                        "json_error": str(e),
                        "phase": "json_recovery"
                    }
                )
                
                match = re.search(r"({.*}|\[.*\])", content, re.DOTALL)
                if match:
                    try:
                        json_data = json.loads(match.group(1))
                        await event_logger.log(
                            EventType.LLM_REQUEST,
                            f"JSON recovery successful with regex extraction",
                            Priority.NORMAL,
                            metadata={
                                "operation": "call_llm_structured",
                                "model": model,
                                "response_model": model_name,
                                "attempt": attempt + 1,
                                "recovery_method": "regex_json",
                                "phase": "json_recovered"
                            }
                        )
                    except json.JSONDecodeError:
                        json_data = dirtyjson.loads(match.group(1))
                        await event_logger.log(
                            EventType.LLM_REQUEST,
                            f"JSON recovery successful with dirty JSON",
                            Priority.NORMAL,
                            metadata={
                                "operation": "call_llm_structured",
                                "model": model,
                                "response_model": model_name,
                                "attempt": attempt + 1,
                                "recovery_method": "dirtyjson",
                                "phase": "json_recovered"
                            }
                        )
                else:
                    try:
                        json_data = dirtyjson.loads(content)
                        await event_logger.log(
                            EventType.LLM_REQUEST,
                            f"JSON recovery successful with direct dirty JSON",
                            Priority.NORMAL,
                            metadata={
                                "operation": "call_llm_structured",
                                "model": model,
                                "response_model": model_name,
                                "attempt": attempt + 1,
                                "recovery_method": "direct_dirtyjson",
                                "phase": "json_recovered"
                            }
                        )
                    except Exception as recovery_error:
                        await event_logger.log(
                            EventType.LLM_REQUEST,
                            f"All JSON recovery attempts failed (attempt {attempt + 1})",
                            Priority.HIGH,
                            metadata={
                                "operation": "call_llm_structured",
                                "model": model,
                                "response_model": model_name,
                                "attempt": attempt + 1,
                                "recovery_error": str(recovery_error),
                                "phase": "json_recovery_failed"
                            }
                        )
                        last_error = ValidationError.from_exception_data(
                            "ValidationError",
                            [
                                {
                                    "type": "json_invalid",
                                    "loc": ("response",),
                                    "input": content,
                                }
                            ],
                        )
                        json_data = None
            if json_data is not None:
                # If the model returned an explicit error payload, surface it
                if isinstance(json_data, dict) and any(
                    key in json_data for key in ("error", "errors", "detail")
                ):
                    msg = (
                        json_data.get("error")
                        or json_data.get("detail")
                        or json_data.get("errors")
                    )
                    raise ValueError(str(msg))

                # Existing light unwrapping for RootModel
                root_field = response_model.model_fields.get("root")
                if root_field:
                    if get_origin(root_field.annotation) is list:
                        if isinstance(json_data, dict):
                            if len(json_data) == 1 and isinstance(
                                next(iter(json_data.values())), list
                            ):
                                json_data = next(iter(json_data.values()))
                            else:
                                json_data = [json_data]
                        elif (
                            isinstance(json_data, list)
                            and len(json_data) == 1
                            and isinstance(json_data[0], dict)
                            and len(json_data[0]) == 1
                            and isinstance(next(iter(json_data[0].values())), list)
                        ):
                            json_data = next(iter(json_data[0].values()))
                    elif root_field.annotation in {str, int} and isinstance(
                        json_data, dict
                    ):
                        for key in ("description", "title", "value"):
                            val = json_data.get(key)
                            if isinstance(val, root_field.annotation):
                                json_data = val
                                break

                # Model-specific normalization for robustness (with retryable structural errors)
                try:
                    json_data = _maybe_normalize_for_model(response_model, json_data)
                except RetryableValidationError as rve:
                    if attempt < max_retries:
                        # Prepare stricter retry with example
                        stricter_suffix_used = rve.stricter_suffix
                        example_for_retry = rve.example_json
                        error_msg = f"{rve.reason} (expected {rve.expected_root})"
                        structured_prompt = (
                            f"{system_instr}\n{prompt}\n\nSTRICT FORMAT REQUIREMENTS:\n"
                            f"{rve.stricter_suffix}\nExample:\n{rve.example_json}\n"
                            f"Previous attempt failed with: {error_msg}"
                        )
                        # Debug snapshot for tightened prompt
                        try:
                            header = f"model={model}; structured_json_schema=true; tightened_retry=true"
                            await write_debug_snapshot(
                                base_slug=f"llm_structured_{model}",
                                part="prompt",
                                header=header,
                                body=structured_prompt or "",
                            )
                        except Exception:
                            pass
                        continue
                    else:
                        # No retries left; surface as validation error
                        last_error = ValidationError.from_exception_data(
                            "ValidationError",
                            [
                                {
                                    "type": "structural_mismatch",
                                    "loc": ("response",),
                                    "input": json_data,
                                    "msg": rve.reason,
                                }
                            ],
                        )
                        json_data = None

                if json_data is not None:
                    await event_logger.log(
                        EventType.LLM_REQUEST,
                        f"Validating parsed data against {model_name} schema (attempt {attempt + 1})",
                        Priority.NORMAL,
                        metadata={
                            "operation": "call_llm_structured",
                            "model": model,
                            "response_model": model_name,
                            "attempt": attempt + 1,
                            "phase": "schema_validation"
                        }
                    )
                    
                    try:
                        validated_result = response_model.model_validate(json_data)
                        total_duration = time.time() - start_time
                        
                        await event_logger.log(
                            EventType.LLM_REQUEST,
                            f"Successfully completed structured LLM call to {model} for {model_name}",
                            Priority.NORMAL,
                            metadata={
                                "operation": "call_llm_structured",
                                "model": model,
                                "response_model": model_name,
                                "total_duration": total_duration,
                                "final_attempt": attempt + 1,
                                "success": True,
                                "phase": "completion"
                            }
                        )
                        return validated_result
                        
                    except ValidationError as e:
                        await event_logger.log(
                            EventType.LLM_REQUEST,
                            f"Schema validation failed, attempting normalization (attempt {attempt + 1})",
                            Priority.NORMAL,
                            metadata={
                                "operation": "call_llm_structured",
                                "model": model,
                                "response_model": model_name,
                                "attempt": attempt + 1,
                                "validation_error": str(e),
                                "phase": "validation_failed"
                            }
                        )
                        
                        # One more pass after normalization
                        try:
                            json_data = _maybe_normalize_for_model(
                                response_model, json_data
                            )
                            validated_result = response_model.model_validate(json_data)
                            total_duration = time.time() - start_time
                            
                            await event_logger.log(
                                EventType.LLM_REQUEST,
                                f"Successfully validated after normalization (attempt {attempt + 1})",
                                Priority.NORMAL,
                                metadata={
                                    "operation": "call_llm_structured",
                                    "model": model,
                                    "response_model": model_name,
                                    "total_duration": total_duration,
                                    "final_attempt": attempt + 1,
                                    "normalized": True,
                                    "success": True,
                                    "phase": "completion"
                                }
                            )
                            return validated_result
                            
                        except RetryableValidationError as rve2:
                            if attempt < max_retries:
                                await event_logger.log(
                                    EventType.LLM_REQUEST,
                                    f"Retryable validation error, preparing stricter retry (attempt {attempt + 1})",
                                    Priority.NORMAL,
                                    metadata={
                                        "operation": "call_llm_structured",
                                        "model": model,
                                        "response_model": model_name,
                                        "attempt": attempt + 1,
                                        "retry_reason": rve2.reason,
                                        "expected_root": rve2.expected_root,
                                        "phase": "prepare_retry"
                                    }
                                )
                                
                                stricter_suffix_used = rve2.stricter_suffix
                                example_for_retry = rve2.example_json
                                error_msg = (
                                    f"{rve2.reason} (expected {rve2.expected_root})"
                                )
                                structured_prompt = (
                                    f"{system_instr}\n{prompt}\n\nSTRICT FORMAT REQUIREMENTS:\n"
                                    f"{rve2.stricter_suffix}\nExample:\n{rve2.example_json}\n"
                                    f"Previous attempt failed with: {error_msg}"
                                )
                                try:
                                    header = f"model={model}; structured_json_schema=true; tightened_retry=true"
                                    await write_debug_snapshot(
                                        base_slug=f"llm_structured_{model}",
                                        part="prompt",
                                        header=header,
                                        body=structured_prompt or "",
                                    )
                                except Exception:
                                    pass
                                continue
                            else:
                                last_error = e
                        except ValidationError as e2:
                            last_error = e2

            # Add validation error feedback for retry
            if attempt < max_retries:
                error_msg = str(last_error)
                if stricter_suffix_used:
                    structured_prompt = (
                        f"{system_instr}\n{prompt}\n\nSTRICT FORMAT REQUIREMENTS:\n"
                        f"{stricter_suffix_used}\n"
                        f"{'Example:\n' + example_for_retry if example_for_retry else ''}\n"
                        f"Previous attempt failed with: {error_msg}"
                    )
                else:
                    structured_prompt = f"{system_instr}\n{prompt}\nPrevious attempt failed with error: {error_msg}"

        except Exception as e:
            attempt_duration = time.time() - attempt_start
            await event_logger.log_error_handling_start(
                error_type=type(e).__name__,
                error_msg=str(e),
                context=f"Structured LLM call to {model} for {model_name} (attempt {attempt + 1})",
                metadata={
                    "operation": "call_llm_structured",
                    "model": model,
                    "response_model": model_name,
                    "attempt": attempt + 1,
                    "attempt_duration": attempt_duration,
                    "total_duration": time.time() - start_time
                }
            )
            
            last_error = e
            if attempt < max_retries:
                await event_logger.log(
                    EventType.LLM_REQUEST,
                    f"Attempt {attempt + 1} failed, retrying structured LLM call",
                    Priority.NORMAL,
                    metadata={
                        "operation": "call_llm_structured",
                        "model": model,
                        "response_model": model_name,
                        "attempt": attempt + 1,
                        "will_retry": True,
                        "phase": "retry_decision"
                    }
                )
                continue
            else:
                await event_logger.log(
                    EventType.LLM_REQUEST,
                    f"All attempts exhausted for structured LLM call to {model}",
                    Priority.HIGH,
                    metadata={
                        "operation": "call_llm_structured",
                        "model": model,
                        "response_model": model_name,
                        "total_attempts": max_retries + 1,
                        "total_duration": time.time() - start_time,
                        "final_error": str(e),
                        "phase": "final_failure"
                    }
                )
                raise

    # If we get here, all retries failed
    total_duration = time.time() - start_time
    await event_logger.log(
        EventType.LLM_REQUEST,
        f"Structured LLM call to {model} failed after all retries",
        Priority.HIGH,
        metadata={
            "operation": "call_llm_structured",
            "model": model,
            "response_model": model_name,
            "total_duration": total_duration,
            "total_attempts": max_retries + 1,
            "success": False,
            "phase": "final_failure"
        }
    )
    
    if last_error:
        raise last_error
    else:
        raise RuntimeError("Unexpected error in structured LLM call")


async def embed_text(model: str, text: str) -> list[float]:
    """Return the embedding vector for ``text``.

    Parameters
    ----------
    model:
        Name of the embedding model to query.
    text:
        Text to embed.

    Returns
    -------
    list[float]
        The embedding returned by the provider.

    Raises
    ------
    RuntimeError
        If required environment variables are missing.
    """
    
    start_time = time.time()
    text_length = len(text) if text else 0
    
    await event_logger.log(
        EventType.LLM_REQUEST,
        f"Starting embedding generation with {model}",
        Priority.NORMAL,
        metadata={
            "operation": "embed_text",
            "model": model,
            "text_length": text_length,
            "embedding": True
        }
    )

    import litellm

    api_base = config.embedding.api_base or config.llm.api_base
    api_key = config.embedding.api_key or config.llm.api_key
    if not api_base or not api_key:
        await event_logger.log_error_handling_start(
            error_type="RuntimeError",
            error_msg="EMBEDDING_API_BASE/KEY or OPENAI_API_BASE/KEY must be set",
            context=f"Embedding generation with {model}",
            metadata={
                "operation": "embed_text",
                "model": model,
                "error_category": "configuration",
                "text_length": text_length
            }
        )
        raise RuntimeError("EMBEDDING_API_BASE/KEY or OPENAI_API_BASE/KEY must be set")

    try:
        await event_logger.log(
            EventType.LLM_REQUEST,
            f"Sending embedding request to {model}",
            Priority.NORMAL,
            metadata={
                "operation": "embed_text",
                "model": model,
                "api_base": api_base,
                "text_length": text_length,
                "phase": "request_send"
            }
        )
        
        response: dict[str, Any] = await litellm.aembedding(
            model=model,
            input=text,
            api_base=api_base,
            api_key=api_key,
        )
        
        embedding = response["data"][0]["embedding"]
        embedding_dimensions = len(embedding) if embedding else 0
        duration = time.time() - start_time
        
        await event_logger.log(
            EventType.LLM_REQUEST,
            f"Successfully generated embedding with {model}",
            Priority.NORMAL,
            metadata={
                "operation": "embed_text",
                "model": model,
                "duration": duration,
                "text_length": text_length,
                "embedding_dimensions": embedding_dimensions,
                "success": True
            }
        )
        
        return embedding
        
    except Exception as e:
        duration = time.time() - start_time
        await event_logger.log_error_handling_start(
            error_type=type(e).__name__,
            error_msg=str(e),
            context=f"Embedding generation with {model}",
            metadata={
                "operation": "embed_text",
                "model": model,
                "duration": duration,
                "text_length": text_length
            }
        )
        raise
