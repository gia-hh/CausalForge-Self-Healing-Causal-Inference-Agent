"""Node 8 — Report Node, and the human-escalation terminal node."""

from __future__ import annotations

from datetime import datetime

from ..llm import call_light
from ..llm.prompts import REPORT_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.validators import validate_report


def report_node(state: AgentState) -> dict:
    """Generate the final Markdown report (only reached when sanity passed)."""
    results = state.get("execution_results", {})
    design = state.get("experiment_design", {})
    roles = design.get("variable_roles", {})
    hte = state.get("hte_results", {})

    prompt = REPORT_SYSTEM_PROMPT.format(
        query=state.get("query", ""),
        method=results.get("method", design.get("method", "")),
        data_type=design.get("data_type", ""),
        outcome_type=roles.get("outcome_type", ""),
        ate=results.get("ate"),
        p_value=results.get("p_value"),
        ci=results.get("confidence_interval"),
        n=results.get("sample_size"),
        hte_interpretation=hte,
        sanity_check_details=state.get("sanity_check_details", []),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    report = call_light("report", prompt)

    ok, reason = validate_report(report)
    if not ok:
        print(f"[report] validation warning: {reason}")

    return {"final_report": report}


def human_escalation_node(state: AgentState) -> dict:
    """Terminal node: print why the run halted and stop."""
    attempts = state.get("repair_attempts", 0)
    error = state.get("latest_execution_error", {})
    if not state.get("sanity_check_passed", True) and state.get("sanity_check_details"):
        reason = "sanity check failed"
    elif attempts:
        reason = f"repair attempts exhausted ({attempts})"
    else:
        reason = "escalated to human"

    print("\n" + "=" * 60)
    print(f"[HUMAN ESCALATION] {reason}")
    if error:
        print(f"  last error: {error.get('error_type')} -> {error.get('traceback')}")
    print("=" * 60 + "\n")
    return {"halt_reason": reason}
