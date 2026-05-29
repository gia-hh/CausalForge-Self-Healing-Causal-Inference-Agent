"""
Deterministic, offline "mock" LLM backend.

The mock returns hand-authored responses keyed by a routing tag so the entire
LangGraph pipeline runs end-to-end with no network and reproducible output. It is
used when LLM_BACKEND="mock".

The dispatcher (client.py) tags each call with a ``role`` so the mock knows which
canned response to return. The mock is intentionally "correct" for the happy path;
the fault-injection demos in the runner deliberately corrupt the generated code to
exercise the repair loop.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Mock experiment design (Planning Node)
# ---------------------------------------------------------------------------
_MOCK_EXPERIMENT_DESIGN = {
    "data_type": "observational",
    "variable_roles": {
        "treatment": "coupon_assigned",
        "outcome": "ctr_click",
        "outcome_type": "binary",
        "confounders": ["historical_activity"],
        "unclear_variables": [],
    },
    "dag": {
        "nodes": ["coupon_assigned", "ctr_click", "historical_activity"],
        "edges": [
            {"from": "historical_activity", "to": "coupon_assigned"},
            {"from": "historical_activity", "to": "ctr_click"},
            {"from": "coupon_assigned", "to": "ctr_click"},
        ],
    },
    "method": "PSM",
    "method_rationale": (
        "Observational data with selection bias and no time dimension; "
        "propensity-score IPW controls the confounder."
    ),
    "key_assumptions": [
        "Conditional ignorability given historical_activity",
        "Positivity / overlap holds (propensity bounded away from 0 and 1)",
        "historical_activity is a pre-treatment variable",
    ],
}


# ---------------------------------------------------------------------------
# Mock generated analysis code (CodeGen Node)
# ---------------------------------------------------------------------------
# A complete, correct IPW + LPM implementation honoring the v2 output contract.
# This code is what the Executor runs; it computes the overall ATE, stratified
# ATEs with the SAME IPW weights, exposes propensity_scores and model_fit.
_MOCK_ANALYSIS_CODE = r'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# --- Variable roles (from the experiment design) ---------------------------
treatment = "coupon_assigned"
outcome = "ctr_click"
confounder = "historical_activity"

T = df[treatment].to_numpy().astype(float)
Y = df[outcome].to_numpy().astype(float)
Xconf = df[[confounder]].to_numpy().astype(float)

# --- 1. Propensity score on the full sample (binary treatment) -------------
ps_model = LogisticRegression()
ps_model.fit(Xconf, T)
ps = ps_model.predict_proba(Xconf)[:, 1]
ps = np.clip(ps, 0.01, 0.99)  # trim to preserve overlap / avoid huge weights

# --- 2. IPW weights --------------------------------------------------------
ipw = T / ps + (1.0 - T) / (1.0 - ps)


def ipw_ate(y, t, w):
    """IPW-weighted mean-difference ATE via a weighted LPM on a constant + T."""
    X = sm.add_constant(t)  # columns: [const, T]
    fit = sm.WLS(y, X, weights=w).fit(cov_type="HC3")
    ate = float(fit.params[1])
    p_value = float(fit.pvalues[1])
    ci_low, ci_high = fit.conf_int()[1]
    return ate, p_value, [float(ci_low), float(ci_high)], fit


# --- 3. Overall ATE (full-sample IPW) --------------------------------------
ate, p_value, ci, model_fit = ipw_ate(Y, T, ipw)

# --- 4. Stratified ATE: split by confounder median, reuse the SAME weights --
median = float(np.median(Xconf[:, 0]))
high_mask = Xconf[:, 0] >= median
low_mask = ~high_mask

ate_h, p_h, _, _ = ipw_ate(Y[high_mask], T[high_mask], ipw[high_mask])
ate_l, p_l, _, _ = ipw_ate(Y[low_mask], T[low_mask], ipw[low_mask])

stratified_results = {
    f"{confounder}_high": {"ate": float(ate_h), "p_value": float(p_h),
                            "n": int(high_mask.sum())},
    f"{confounder}_low": {"ate": float(ate_l), "p_value": float(p_l),
                           "n": int(low_mask.sum())},
}

# --- 5. Assemble the mandatory results_dict --------------------------------
results_dict = {
    "method": "PSM-IPW (LPM)",
    "ate": float(ate),
    "p_value": float(p_value),
    "confidence_interval": ci,
    "sample_size": int(len(df)),
    "stratified_results": stratified_results,
    "propensity_scores": ps.tolist(),
}

print(f"[analysis] overall ATE={ate:.4f} p={p_value:.4f} CI={ci}")
print(f"[analysis] high ATE={ate_h:.4f} low ATE={ate_l:.4f}")
'''


def _mock_planning() -> str:
    return json.dumps(_MOCK_EXPERIMENT_DESIGN)


def _mock_codegen() -> str:
    return _MOCK_ANALYSIS_CODE


def _mock_repair(prompt: str) -> str:
    """Mock repair: return a clean, correct copy of the analysis code.

    The fault-injection demos corrupt the code before execution; the mock repair
    "fixes" it by returning the known-good implementation. To honor the blacklist
    self-evaluation (which rejects an identical-to-original copy), we append a
    harmless unique comment so the repaired text differs from any corrupted input.
    """
    return _MOCK_ANALYSIS_CODE + "\n# repaired by mock backend\n"


def _mock_parser() -> str:
    return json.dumps(
        {
            "error_type": "RuntimeError",
            "error_line": -1,
            "code_snippet": "unknown",
            "semantic_summary": "Injected fault detected; regenerating clean code.",
            "approach_tried": "corrupted reference implementation",
        }
    )


def _mock_hte(prompt: str) -> str:
    return json.dumps(
        {
            "highest_effect_segment": (
                "High historical-activity users show the largest lift "
                "(~6 pp), consistent with the DGP."
            ),
            "lowest_effect_segment": (
                "Low historical-activity users show a smaller lift (~3 pp)."
            ),
            "business_interpretation": (
                "Coupons move already-engaged users the most: they were closer "
                "to clicking and the incentive pushes them over the line. Less "
                "active users respond too, but with a weaker marginal effect."
            ),
            "recommendations": [
                "Prioritize coupon spend on high-activity segments for the best ROI.",
                "Test richer incentives for low-activity users before scaling.",
            ],
        }
    )


def _mock_sanity() -> str:
    # Returns passed=true with all six checks; the host also re-verifies numerically.
    return json.dumps(
        {
            "passed": True,
            "checks": [
                {"item": "p_value in [0,1]", "result": "pass", "detail": "ok"},
                {"item": "|ATE| < 1.0", "result": "pass", "detail": "ok"},
                {"item": "CI lower < upper", "result": "pass", "detail": "ok"},
                {"item": "sample size >= 500", "result": "pass", "detail": "ok"},
                {"item": "strata sign agrees with overall", "result": "pass", "detail": "ok"},
                {"item": "strata mean within +/-30%", "result": "pass", "detail": "ok"},
            ],
            "critical_issues": [],
        }
    )


def _mock_report(prompt: str) -> str:
    return (
        "# Causal Inference Analysis Report\n\n"
        "## 1. Business Problem and Method\n"
        "We estimate the effect of coupon assignment on click-through using "
        "observational data. Because more active users are more likely to receive "
        "a coupon (selection bias), we control the confounder with propensity-score "
        "IPW rather than a naive comparison.\n\n"
        "## 2. Causal Effect Estimate\n"
        "The IPW-weighted Linear Probability Model estimates an ATE of roughly 5 "
        "percentage points, statistically significant at the 5% level. The ATE unit "
        "is percentage points because the outcome is binary.\n\n"
        "## 3. Heterogeneity Analysis\n"
        "High historical-activity users show the largest lift (~6 pp) versus low-"
        "activity users (~3 pp). Coupons move already-engaged users the most.\n\n"
        "## 4. Data Quality and Sanity Checks\n"
        "All six sanity checks passed: p-value in range, |ATE| below 1.0, valid CI, "
        "sample size above 500, strata signs consistent with the overall effect, and "
        "strata mean within tolerance.\n\n"
        "## 5. Conclusions and Business Recommendations\n"
        "Coupons causally raise click-through by about 5 pp, with the strongest "
        "effect among active users. Recommendations: (1) prioritize coupon spend on "
        "high-activity segments; (2) test richer incentives for low-activity users "
        "before scaling.\n\n"
        "---\n"
        "*Report generated at: (mock) | System: Automated Causal Inference v2.0*\n"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def mock_complete(role: str, prompt: str) -> str:
    """Return a deterministic canned response for the given node role."""
    dispatch = {
        "planning": lambda: _mock_planning(),
        "codegen": lambda: _mock_codegen(),
        "repair": lambda: _mock_repair(prompt),
        "parser": lambda: _mock_parser(),
        "hte": lambda: _mock_hte(prompt),
        "sanity": lambda: _mock_sanity(),
        "report": lambda: _mock_report(prompt),
    }
    if role not in dispatch:
        raise ValueError(f"Unknown mock role: {role}")
    return dispatch[role]()
