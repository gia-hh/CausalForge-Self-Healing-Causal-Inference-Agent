"""Node 5 — Repair Node (heavy LLM; blacklist-aware, bounded context)."""

from __future__ import annotations

from ..llm import call_heavy
from ..llm.prompts import REPAIR_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.json_utils import strip_code_fences
from ..utils.repair_utils import build_repair_context
from ..utils.validators import validate_repair


def repair_node(state: AgentState) -> dict:
    """Produce a repaired version of the latest code; increment repair_attempts.

    Returns ``repair_attempts: 1`` (the additive reducer accumulates the count).
    Context is bounded: only the latest code, latest parsed error, and a blacklist
    summary are sent — never the full history of prior code versions.
    """
    latest_code = state.get("latest_code", "")
    parsed_error = state.get("latest_parsed_error", {})
    blacklist = state.get("error_blacklist", [])

    ctx = build_repair_context(latest_code, parsed_error, blacklist)
    prompt = REPAIR_SYSTEM_PROMPT.format(**ctx)

    response = call_heavy("repair", prompt)
    repaired = strip_code_fences(response)

    ok, reason = validate_repair(latest_code, repaired, blacklist)
    if not ok:
        # Keep the new code but annotate the failure; the next Executor pass will
        # re-trigger parsing/repair, and the attempt counter still advances so the
        # circuit breaker remains effective.
        print(f"[repair] self-evaluation warning: {reason}")

    return {"latest_code": repaired, "repair_attempts": 1}
