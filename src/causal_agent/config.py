"""
Central configuration for the Automated Causal Inference Expert System.

All tunable constants, model identifiers, and backend switches live here so that
swapping models or toggling offline/mock execution requires no code changes
elsewhere.

Environment variables
---------------------
LLM_BACKEND        : "mock" (default) | "real"
                     "mock"  -> deterministic offline responses, no network.
                     "real"  -> Anthropic API for heavy nodes, Ollama for light nodes.
ANTHROPIC_API_KEY  : required only when LLM_BACKEND="real" (for heavy nodes).
OLLAMA_HOST        : Ollama base URL, default "http://localhost:11434".
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# "mock" lets the entire pipeline run offline end-to-end with deterministic,
# hand-authored LLM responses. "real" routes heavy nodes to the Anthropic API
# and light nodes to a local Ollama server.
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "mock").lower()

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
# NOTE: The original framework named "claude-sonnet-4" and "claude-haiku-4".
# Those API strings are retired / nonexistent. We use the current valid IDs.
# Heavy reasoning (Planning / CodeGen / Repair) -> Anthropic Sonnet.
# Light tasks (Parser / HTE / Sanity / Report)  -> local Ollama Llama 3 8B.
HEAVY_MODEL: str = os.environ.get("HEAVY_MODEL", "claude-sonnet-4-6")
LIGHT_MODEL_OLLAMA: str = os.environ.get("LIGHT_MODEL_OLLAMA", "llama3:8b")

# Anthropic API settings (used only when LLM_BACKEND="real")
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MAX_TOKENS: int = 4096
ANTHROPIC_TIMEOUT_S: float = 120.0

# Ollama settings (used only when LLM_BACKEND="real")
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT_S: float = 120.0

# ---------------------------------------------------------------------------
# Synthetic data ground truth
# ---------------------------------------------------------------------------
SYNTH_N: int = 5000
SYNTH_SEED: int = 42
TRUE_ATE: float = 0.05  # 5 percentage-point lift (god's-eye ground truth)

# Acceptance bounds for ATE recovery on the synthetic dataset.
ATE_LOWER_BOUND: float = 0.04
ATE_UPPER_BOUND: float = 0.06
P_VALUE_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Repair / circuit-breaker
# ---------------------------------------------------------------------------
# Maximum number of repair attempts before escalating to a human.
# The router escalates once repair_attempts >= MAX_REPAIR_ATTEMPTS, which
# yields exactly MAX_REPAIR_ATTEMPTS repairs (each repair adds 1).
MAX_REPAIR_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Statistical diagnostic thresholds (host-side, enforced by Executor)
# ---------------------------------------------------------------------------
VIF_THRESHOLD: float = 10.0
BP_PVALUE_THRESHOLD: float = 0.01

# ---------------------------------------------------------------------------
# Sanity-check thresholds
# ---------------------------------------------------------------------------
MIN_SAMPLE_SIZE: int = 500
HTE_CONSISTENCY_TOLERANCE: float = 0.30  # allowed relative deviation


def is_mock() -> bool:
    """Return True when running with the deterministic offline backend."""
    return LLM_BACKEND == "mock"


def describe_backend() -> str:
    """Human-readable one-line description of the active backend."""
    if is_mock():
        return "MOCK (deterministic offline; no network calls)"
    return (
        f"REAL (heavy={HEAVY_MODEL} via Anthropic API; "
        f"light={LIGHT_MODEL_OLLAMA} via Ollama @ {OLLAMA_HOST})"
    )
