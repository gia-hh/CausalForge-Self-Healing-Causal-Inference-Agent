"""Node 1 — Planning Node (heavy LLM + LangGraph interrupt HITL)."""

from __future__ import annotations

from langgraph.types import interrupt

from ..llm import call_heavy
from ..llm.prompts import PLANNING_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.json_utils import parse_json_response
from ..utils.validators import validate_experiment_design


def planning_node(state: AgentState) -> dict:
    """Build the experiment design; pause for human input on unclear variables."""
    prompt = PLANNING_SYSTEM_PROMPT.format(
        query=state["query"], metadata=state["metadata"]
    )
    response = call_heavy("planning", prompt)
    experiment_design = parse_json_response(response)

    if not validate_experiment_design(experiment_design):
        raise ValueError("experiment_design validation failed; check LLM output")

    # HITL: non-blocking interrupt on uncertain variable roles.
    unclear = experiment_design["variable_roles"].get("unclear_variables", [])
    if unclear:
        # interrupt() pauses execution and returns control to the caller, which
        # later resumes with Command(resume=decisions).
        human_decisions = interrupt(
            {
                "type": "confounder_confirmation",
                "unclear_variables": unclear,
                "instruction": (
                    "For each variable reply y (is a confounder) or n (is not)."
                ),
            }
        )
        confirmed = [
            v for v in unclear if str(human_decisions.get(v, "n")).lower() == "y"
        ]
        experiment_design["variable_roles"]["confounders"].extend(confirmed)
        experiment_design["variable_roles"]["unclear_variables"] = []

    return {"experiment_design": experiment_design}
