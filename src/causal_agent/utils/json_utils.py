"""Helpers for extracting JSON from (possibly noisy) LLM text responses."""

from __future__ import annotations

import json
import re


def parse_json_response(raw: str) -> dict:
    """Parse a JSON object from an LLM response.

    Tolerates markdown code fences and leading/trailing prose by extracting the
    outermost ``{...}`` block when a direct parse fails.

    Raises:
        ValueError: if no valid JSON object can be recovered.
    """
    text = raw.strip()

    # Strip markdown fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Fast path.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first balanced-looking {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from LLM response: {exc}") from exc

    raise ValueError("No JSON object found in LLM response.")


def strip_code_fences(raw: str) -> str:
    """Return code with surrounding markdown ``` fences removed, if any."""
    text = raw.strip()
    fence = re.match(r"^```(?:python)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text
