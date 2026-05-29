"""
Unified LLM client.

Exposes two role-aware entry points used by the nodes:

    call_heavy(role, system_prompt) -> str     # Planning / CodeGen / Repair
    call_light(role, system_prompt) -> str     # Parser / HTE / Sanity / Report

Routing
-------
* LLM_BACKEND="mock": both helpers return deterministic canned text.
* LLM_BACKEND="real":
    - heavy -> Anthropic Messages API (config.HEAVY_MODEL)
    - light -> local Ollama (config.LIGHT_MODEL_OLLAMA)

The ``role`` tag is what the mock backend keys on; for real backends it is used
only for logging.
"""

from __future__ import annotations

import json
import urllib.request

from .. import config
from .mock_client import mock_complete

HEAVY_ROLES = {"planning", "codegen", "repair"}
LIGHT_ROLES = {"parser", "hte", "sanity", "report"}


# ---------------------------------------------------------------------------
# Real backends
# ---------------------------------------------------------------------------
def _call_anthropic(system_prompt: str) -> str:
    """Call the Anthropic Messages API. Requires the `anthropic` package + key."""
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set but LLM_BACKEND=real requires it for "
            "heavy nodes. Set the key or use LLM_BACKEND=mock."
        )
    try:
        import anthropic  # imported lazily so mock mode needs no dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'anthropic' package is required for LLM_BACKEND=real. "
            "Install it with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY, timeout=config.ANTHROPIC_TIMEOUT_S
    )
    # The whole instruction (including data) is delivered as the single user
    # message; we keep the system slot generic to avoid duplicating content.
    resp = client.messages.create(
        model=config.HEAVY_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
        messages=[{"role": "user", "content": system_prompt}],
    )
    # Concatenate all text blocks.
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()


def _call_ollama(system_prompt: str) -> str:
    """Call a local Ollama server's /api/generate endpoint (no extra deps)."""
    url = f"{config.OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = json.dumps(
        {
            "model": config.LIGHT_MODEL_OLLAMA,
            "prompt": system_prompt,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT_S) as r:
        body = json.loads(r.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def call_heavy(role: str, system_prompt: str) -> str:
    """Heavy-reasoning call (Planning / CodeGen / Repair)."""
    if role not in HEAVY_ROLES:
        raise ValueError(f"'{role}' is not a heavy role: {HEAVY_ROLES}")
    if config.is_mock():
        return mock_complete(role, system_prompt)
    return _call_anthropic(system_prompt)


def call_light(role: str, system_prompt: str) -> str:
    """Light-reasoning call (Parser / HTE / Sanity / Report)."""
    if role not in LIGHT_ROLES:
        raise ValueError(f"'{role}' is not a light role: {LIGHT_ROLES}")
    if config.is_mock():
        return mock_complete(role, system_prompt)
    return _call_ollama(system_prompt)
