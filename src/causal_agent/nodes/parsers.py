"""Nodes 4a/4b — Parsers, and the conditional-edge router functions."""

from __future__ import annotations

from .. import config
from ..llm import call_light
from ..llm.prompts import LLM_PARSER_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.repair_utils import (
    build_blacklist_entry,
    rule_based_parser,
    validate_and_parse_llm_output,
)

# Error classes routed to the deterministic rule-based parser.
_SYNTAX_CLASS = {
    "SyntaxError",
    "IndentationError",
    "NameError",
    "ImportError",
    "AttributeError",
    "OutputConventionError",
    "ResultValidationError",
}


# ---------------------------------------------------------------------------
# Routers (conditional edges)
# ---------------------------------------------------------------------------
def execution_router(state: AgentState) -> str:
    """Route after the Executor: success -> HTE; failure -> parser or escalation."""
    error = state.get("latest_execution_error") or {}
    # No error recorded -> clean execution.
    if not error:
        return "hte_supervisor"

    if state.get("repair_attempts", 0) >= config.MAX_REPAIR_ATTEMPTS:
        return "human_escalation"

    error_type = error.get("error_type", "")
    if error_type in _SYNTAX_CLASS:
        return "rule_based_parser"
    # StatisticalAssumptionError, RuntimeError, etc. need semantic understanding.
    return "llm_parser"


def sanity_router(state: AgentState) -> str:
    """Route after Sanity Check: passed -> report; failed -> escalation."""
    return "report" if state.get("sanity_check_passed") else "human_escalation"


# ---------------------------------------------------------------------------
# Node 4a — rule-based parser
# ---------------------------------------------------------------------------
def rule_based_parser_node(state: AgentState) -> dict:
    """Parse syntactic/structural errors deterministically; append to blacklist."""
    error = state.get("latest_execution_error") or {}
    parsed = rule_based_parser(state.get("latest_code", ""), error)
    return {
        "latest_parsed_error": parsed,
        "error_blacklist": [build_blacklist_entry(parsed)],
    }


# ---------------------------------------------------------------------------
# Node 4b — LLM parser
# ---------------------------------------------------------------------------
def llm_parser_node(state: AgentState) -> dict:
    """Parse semantic errors via the light LLM; fall back to the rule parser."""
    error = state.get("latest_execution_error") or {}
    code = state.get("latest_code", "")
    prompt = LLM_PARSER_SYSTEM_PROMPT.format(
        code=code, traceback=error.get("traceback", "")
    )
    raw = call_light("parser", prompt)
    ok, parsed = validate_and_parse_llm_output(raw)
    if not ok:
        parsed = rule_based_parser(code, error)
    return {
        "latest_parsed_error": parsed,
        "error_blacklist": [build_blacklist_entry(parsed)],
    }
