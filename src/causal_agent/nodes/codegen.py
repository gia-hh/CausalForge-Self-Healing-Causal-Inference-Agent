"""Node 2 — CodeGen Node (heavy LLM)."""

from __future__ import annotations

import os

from ..llm import call_heavy
from ..llm.prompts import CODEGEN_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.json_utils import strip_code_fences
from ..utils.validators import pre_check_code


def _maybe_inject_fault(code: str) -> str:
    """Optionally corrupt the generated code to exercise the repair loop.

    Controlled by the INJECT_FAULT env var (used by demos / acceptance tests):
      - "nameerror": reference an undefined variable -> NameError (rule parser path)
      - "multicollinearity": add a duplicated regressor -> VIF>10
        StatisticalAssumptionError (LLM parser path)
    The pre-check still passes because the mandatory tokens remain present.
    """
    fault = os.environ.get("INJECT_FAULT", "").lower()
    if fault == "nameerror":
        # Insert a reference to an undefined name before results_dict is built.
        marker = "# --- 5. Assemble"
        if marker in code:
            return code.replace(
                marker,
                "ate = ate + undefined_variable_injected_fault  # injected NameError\n"
                + marker,
                1,
            )
    elif fault == "multicollinearity":
        # Refit model_fit with a duplicated (perfectly collinear) regressor so the
        # host VIF check trips. Appended after the original code so tokens remain.
        return code + (
            "\n# --- injected multicollinearity fault ---\n"
            "import numpy as _np_fault\n"
            "import statsmodels.api as _sm_fault\n"
            "_dup = Xconf[:, 0]\n"
            "_Xbad = _np_fault.column_stack([T, Xconf[:, 0], _dup])\n"
            "_Xbad = _sm_fault.add_constant(_Xbad)\n"
            "model_fit = _sm_fault.WLS(Y, _Xbad, weights=ipw).fit()\n"
        )
    return code


def codegen_node(state: AgentState) -> dict:
    """Generate the initial analysis code from the experiment design."""
    prompt = CODEGEN_SYSTEM_PROMPT.format(
        experiment_design=state["experiment_design"],
        metadata=state["metadata"],
    )
    response = call_heavy("codegen", prompt)
    code = strip_code_fences(response)

    ok, reason = pre_check_code(code)
    if not ok:
        # A failed static pre-check is itself an error to be repaired; surface it
        # as an execution error so the router sends it to the parser/repair loop.
        return {
            "latest_code": code,
            "latest_execution_error": {
                "error_type": "OutputConventionError",
                "traceback": f"static pre-check failed: {reason}",
            },
        }

    code = _maybe_inject_fault(code)
    return {"latest_code": code}
