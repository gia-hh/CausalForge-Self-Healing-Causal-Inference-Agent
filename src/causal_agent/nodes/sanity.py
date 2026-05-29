"""Node 7 — Sanity Check Node (light LLM + host-side numeric re-verification)."""

from __future__ import annotations

from .. import config
from ..llm import call_light
from ..llm.prompts import SANITY_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.json_utils import parse_json_response
from ..utils.validators import validate_hte_consistency


def _host_checks(results: dict) -> tuple[bool, list]:
    """Deterministic re-verification of the six sanity items (does not trust LLM)."""
    checks = []
    passed = True

    def record(item: str, ok: bool, detail: str):
        nonlocal passed
        checks.append({"item": item, "result": "pass" if ok else "fail", "detail": detail})
        passed = passed and ok

    ate = results.get("ate")
    p = results.get("p_value")
    ci = results.get("confidence_interval", [None, None])
    n = results.get("sample_size", 0)
    strats = results.get("stratified_results", {})

    record("p_value in [0,1]", isinstance(p, (int, float)) and 0.0 <= p <= 1.0, f"p={p}")
    record("|ATE| < 1.0", isinstance(ate, (int, float)) and abs(ate) < 1.0, f"ATE={ate}")
    record(
        "CI lower < upper",
        isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None
        and ci[1] is not None and ci[0] < ci[1],
        f"CI={ci}",
    )
    record("sample size >= 500", isinstance(n, int) and n >= config.MIN_SAMPLE_SIZE, f"n={n}")

    sign_ok = True
    if isinstance(ate, (int, float)) and strats:
        for v in strats.values():
            if ate * v.get("ate", 0) < 0:
                sign_ok = False
    record("strata signs agree with overall", sign_ok, "all strata same sign as ATE")

    consistency_ok, consistency_detail = validate_hte_consistency(results)
    record("strata mean within +/-30%", consistency_ok, consistency_detail)

    return passed, checks


def sanity_check_node(state: AgentState) -> dict:
    """Run the LLM checklist, then re-verify numerically on the host (host wins)."""
    results = state.get("execution_results", {})
    hte = state.get("hte_results", {})
    design = state.get("experiment_design", {})

    prompt = SANITY_SYSTEM_PROMPT.format(
        execution_results=results,
        hte_results=hte,
        experiment_design=design,
    )
    raw = call_light("sanity", prompt)
    try:
        llm_view = parse_json_response(raw)
    except ValueError:
        llm_view = {"passed": False, "checks": [], "critical_issues": ["LLM parse failed"]}

    # Host re-verification is authoritative: the result passes only if BOTH the
    # host checks and the LLM agree.
    host_passed, host_checks = _host_checks(results)
    final_passed = bool(host_passed and llm_view.get("passed", False))

    details = host_checks + [{"item": "llm_review", "result": "pass" if llm_view.get("passed") else "fail",
                              "detail": str(llm_view.get("critical_issues", []))}]

    if not final_passed:
        print("[sanity] FAILED. critical issues:")
        for c in details:
            if c["result"] == "fail":
                print(f"  - {c['item']}: {c['detail']}")

    return {"sanity_check_passed": final_passed, "sanity_check_details": details}
