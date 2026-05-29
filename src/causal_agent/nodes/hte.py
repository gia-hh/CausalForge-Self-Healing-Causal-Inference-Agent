"""Node 6 — Supervisor / HTE Node (light LLM; pure interpretation, no stats)."""

from __future__ import annotations

from ..llm import call_light
from ..llm.prompts import HTE_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.json_utils import parse_json_response
from ..utils.validators import validate_hte_consistency


def hte_supervisor_node(state: AgentState) -> dict:
    """Interpret pre-computed stratified ATEs; never recompute statistics here."""
    results = state.get("execution_results", {})

    # Host-side consistency guard (pure Python) before asking the LLM to interpret.
    ok, detail = validate_hte_consistency(results)
    if not ok:
        print(f"[HTE] consistency warning: {detail}")

    design = state.get("experiment_design", {})
    roles = design.get("variable_roles", {})
    confounders = roles.get("confounders", [])
    stratification_variable = confounders[0] if confounders else "(unknown)"

    prompt = HTE_SYSTEM_PROMPT.format(
        ate=results.get("ate"),
        p_value=results.get("p_value"),
        ci=results.get("confidence_interval"),
        stratified_results=results.get("stratified_results"),
        stratification_variable=stratification_variable,
        treatment=roles.get("treatment"),
        outcome=roles.get("outcome"),
    )
    raw = call_light("hte", prompt)
    try:
        hte = parse_json_response(raw)
    except ValueError:
        hte = {
            "highest_effect_segment": "unavailable (interpretation parse failed)",
            "lowest_effect_segment": "unavailable",
            "business_interpretation": "unavailable",
            "recommendations": [],
        }

    return {"hte_results": hte}
