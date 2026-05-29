"""
Pure-Python self-evaluation validators used across nodes.

These are deterministic guards (no LLM) that enforce the v2 contracts: a valid
experiment design, the mandatory code output tokens, sane numeric results, a
genuine (non-identical, non-blacklisted) repair, HTE consistency, and a
well-formed report.
"""

from __future__ import annotations

import numbers
import math
from typing import Tuple

from .. import config


# ---------------------------------------------------------------------------
# Node 1 — experiment design
# ---------------------------------------------------------------------------
def validate_experiment_design(design: dict) -> bool:
    """Validate the structure and basic content of an experiment design."""
    required_keys = {"data_type", "variable_roles", "dag", "method"}
    required_roles = {"treatment", "outcome", "outcome_type", "confounders"}
    valid_methods = {"PSM", "DID", "PSM+DID", "IV"}

    if not required_keys.issubset(design.keys()):
        print(f"[FAIL] missing fields: {required_keys - design.keys()}")
        return False
    roles = design["variable_roles"]
    if not required_roles.issubset(roles.keys()):
        print(f"[FAIL] variable_roles missing: {required_roles - roles.keys()}")
        return False
    if design["method"] not in valid_methods:
        print(f"[FAIL] unknown method: {design['method']}")
        return False
    if design["data_type"] == "observational" and not roles["confounders"]:
        print("[WARN] observational data with no confounder identified -- risky")
        return False
    return True


# ---------------------------------------------------------------------------
# Node 2 — static code pre-check (before execution)
# ---------------------------------------------------------------------------
def pre_check_code(code: str) -> Tuple[bool, str]:
    """Static rule check on generated code prior to sandbox execution."""
    required_tokens = [
        ("results_dict", "missing results_dict output"),
        ("stratified_results", "missing stratified ATE (stratified_results)"),
        ("model_fit", "missing model_fit object (host diagnostics need it)"),
        ("propensity_scores", "missing propensity_scores"),
    ]
    for token, msg in required_tokens:
        if token not in code:
            return False, msg

    forbidden = ["os.", "subprocess", "sys.exit", "open(", "requests."]
    for f in forbidden:
        if f in code:
            return False, f"contains forbidden module/function: {f}"

    # A raw T-test signals the LLM degraded the stratified estimate.
    if "ttest_ind" in code:
        return False, (
            "detected ttest_ind; stratified ATE must use IPW weights, raw "
            "T-test is forbidden"
        )
    return True, "OK"


# ---------------------------------------------------------------------------
# Node 3 — execution result numeric validation
# ---------------------------------------------------------------------------
def _is_number(x) -> bool:
    return isinstance(x, numbers.Real) and math.isfinite(float(x))


def validate_execution_results(results: dict) -> Tuple[bool, str]:
    """Basic numeric sanity validation of results_dict."""
    ate = results.get("ate")
    p = results.get("p_value")
    ci = results.get("confidence_interval", [None, None])
    sr = results.get("stratified_results")

    if not _is_number(ate):
        return False, f"ATE is not a finite number: {ate}"
    if not _is_number(p) or not (0.0 <= float(p) <= 1.0):
        return False, f"p_value out of [0,1]: {p}"
    if (
        not isinstance(ci, (list, tuple))
        or len(ci) != 2
        or ci[0] is None
        or ci[1] is None
        or float(ci[0]) >= float(ci[1])
    ):
        return False, f"abnormal confidence interval: {ci}"
    if abs(float(ate)) > 1.0:
        return False, f"ATE > 1.0; suspected leakage or scaling error: {ate}"
    if not isinstance(sr, dict) or len(sr) < 2:
        return False, "stratified_results missing or fewer than 2 strata"
    return True, "OK"


# ---------------------------------------------------------------------------
# Node 5 — repair validation
# ---------------------------------------------------------------------------
def validate_repair(original: str, repaired: str, blacklist: list) -> Tuple[bool, str]:
    """Ensure the repaired code is a genuine, contract-preserving change."""
    if original.strip() == repaired.strip():
        return False, "repaired code is identical to the original"
    for token in ("results_dict", "stratified_results", "model_fit"):
        if token not in repaired:
            return False, f"repaired code lost {token}"
    if "ttest_ind" in repaired:
        return False, "repaired code introduced a T-test, violating the IPW contract"
    for entry in blacklist:
        approach = entry.get("approach_tried", "")
        keywords = [w for w in approach.split() if len(w) > 3][:3]
        if len(keywords) >= 2 and all(k in repaired for k in keywords):
            return False, f"appears to reuse a failed approach: {approach}"
    return True, "OK"


# ---------------------------------------------------------------------------
# Node 6 — HTE consistency (pure Python)
# ---------------------------------------------------------------------------
def validate_hte_consistency(
    execution_results: dict, tolerance: float = config.HTE_CONSISTENCY_TOLERANCE
) -> Tuple[bool, str]:
    """Check that stratified ATEs are consistent with the overall ATE."""
    overall = execution_results.get("ate")
    strats = execution_results.get("stratified_results", {})
    if not strats or overall is None or abs(float(overall)) < 1e-6:
        return True, "skipped (insufficient data)"

    overall = float(overall)
    avg_stratified = sum(float(v["ate"]) for v in strats.values()) / len(strats)
    rel_error = abs(avg_stratified - overall) / abs(overall)
    if rel_error > tolerance:
        return False, (
            f"stratified ATE mean ({avg_stratified:.4f}) deviates from overall "
            f"ATE ({overall:.4f}) by {rel_error:.1%}, exceeding tolerance"
        )

    for name, v in strats.items():
        if overall * float(v["ate"]) < 0:  # opposite sign
            return False, (
                f"stratum {name} ATE sign opposes the overall ATE; suspected "
                f"data problem"
            )
    return True, "OK"


# ---------------------------------------------------------------------------
# Node 8 — report validation
# ---------------------------------------------------------------------------
def validate_report(report: str) -> Tuple[bool, str]:
    """Validate the required Markdown section structure and minimum length."""
    required_sections = ["## 1.", "## 2.", "## 3.", "## 4.", "## 5."]
    missing = [s for s in required_sections if s not in report]
    if missing:
        return False, f"report missing sections: {missing}"
    if len(report) < 400:
        return False, "report too short (< 400 chars); suspected truncation"
    return True, "OK"
