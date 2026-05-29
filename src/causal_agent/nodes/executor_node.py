"""Node 3 — Executor Node (host-side; wraps the sandbox)."""

from __future__ import annotations

from .. import runtime
from ..executor import execute_in_sandbox
from ..state import AgentState


def executor_node(state: AgentState) -> dict:
    """Run the latest code; write execution_results on success or the raw error."""
    # If CodeGen already flagged a static pre-check failure, pass it through so the
    # router can dispatch to the repair loop without attempting execution.
    pre_error = state.get("latest_execution_error")
    code = state.get("latest_code", "")
    if pre_error and not code:
        return {}

    df = runtime.get_dataframe()
    outcome = execute_in_sandbox(code, df)

    if outcome["success"]:
        # Clear any stale error so the router reads a clean state.
        return {
            "execution_results": outcome["results"],
            "latest_execution_error": {},
        }

    return {
        "latest_execution_error": {
            "error_type": outcome["error_type"],
            "traceback": outcome["traceback"],
        }
    }
