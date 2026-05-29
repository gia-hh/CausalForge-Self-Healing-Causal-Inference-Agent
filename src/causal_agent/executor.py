"""
Executor (host-side Python; no LLM).

Runs generated code in an isolated namespace, then ENFORCES statistical
diagnostics itself rather than trusting the LLM to self-check. The VIF and
Breusch-Pagan tests are run by the host; a violation raises
``StatisticalAssumptionError``, which downstream routing turns into a semantic
repair request.

This separation of duties is the central v2 architectural change: the LLM only
fits a model and assigns it to ``model_fit``; the host owns the diagnostics.
"""

from __future__ import annotations

import contextlib
import io
import traceback

import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config
from .utils.validators import validate_execution_results


class StatisticalAssumptionError(Exception):
    """Raised by the host Executor when a statistical assumption is badly violated."""


def _run_statistical_diagnostics(model_fit) -> None:
    """Run VIF and Breusch-Pagan tests on a statsmodels fit; raise on violation.

    Diagnostics must never crash the host on benign edge cases (singular design,
    shape mismatch), so the numeric machinery is guarded; only a genuine
    assumption violation raises StatisticalAssumptionError.
    """
    model = getattr(model_fit, "model", None)
    exog = getattr(model, "exog", None)

    # --- VIF (multicollinearity) ------------------------------------------
    # Only meaningful with more than two regressors (e.g. const + T + extra).
    if exog is not None and getattr(exog, "ndim", 0) == 2 and exog.shape[1] > 2:
        try:
            vifs = [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]
        except Exception:
            vifs = []
        # A non-finite VIF (inf / nan) signals perfect or near-perfect
        # collinearity — the most severe case — so it must also trigger.
        if vifs:
            non_finite = any(not np.isfinite(v) for v in vifs)
            finite_vifs = [v for v in vifs if np.isfinite(v)]
            max_vif = max(finite_vifs) if finite_vifs else float("inf")
            if non_finite or max_vif > config.VIF_THRESHOLD:
                shown = "inf" if non_finite else f"{max_vif:.2f}"
                raise StatisticalAssumptionError(
                    f"Multicollinearity: max VIF = {shown} "
                    f"(threshold {config.VIF_THRESHOLD}). Consider dropping a "
                    f"highly-correlated feature or using ridge regression."
                )

    # --- Breusch-Pagan (heteroskedasticity) -------------------------------
    # A Linear Probability Model on a binary outcome is heteroskedastic by
    # construction, so BP will almost always reject. The correct remedy is
    # robust (HC) standard errors. Therefore, if the fit already uses an HC
    # covariance, the assumption has been handled and we do NOT raise. We only
    # raise for a non-robust fit, which is exactly the case the error addresses.
    cov_type = str(getattr(model_fit, "cov_type", "nonrobust")).upper()
    uses_robust_se = cov_type.startswith("HC")
    resid = getattr(model_fit, "resid", None)
    if not uses_robust_se and resid is not None and exog is not None:
        try:
            resid_arr = np.asarray(resid, dtype=float)
            exog_arr = np.asarray(exog, dtype=float)
            if resid_arr.shape[0] == exog_arr.shape[0]:
                _, bp_pvalue, _, _ = het_breuschpagan(resid_arr, exog_arr)
                if np.isfinite(bp_pvalue) and bp_pvalue < config.BP_PVALUE_THRESHOLD:
                    raise StatisticalAssumptionError(
                        f"Heteroskedastic residuals: BP test p-value = "
                        f"{bp_pvalue:.4f}. Use HC3 robust standard errors "
                        f"(sm.OLS(...).fit(cov_type='HC3'))."
                    )
        except StatisticalAssumptionError:
            raise
        except Exception:
            # BP can fail on degenerate designs; never crash the host on it.
            pass


def execute_in_sandbox(code: str, df) -> dict:
    """Execute generated code and enforce host-side diagnostics.

    Steps:
      1. exec() the code in an isolated namespace (df pre-injected).
      2. Extract model_fit -> run VIF / BP diagnostics (host-enforced).
      3. Extract results_dict -> numeric validation.

    Returns a dict; ``success`` indicates whether downstream nodes should treat
    this as a clean execution.
    """
    isolated_ns = {"df": df}
    stdout_capture = io.StringIO()

    # --- Step 1: run the code ---------------------------------------------
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, isolated_ns)  # noqa: S102 - intentional sandboxed exec
    except Exception as e:  # noqa: BLE001 - we surface any failure to the router
        return {
            "success": False,
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }

    # --- Step 2: host-enforced statistical diagnostics --------------------
    model_fit = isolated_ns.get("model_fit")
    if model_fit is not None:
        try:
            _run_statistical_diagnostics(model_fit)
        except StatisticalAssumptionError as e:
            return {
                "success": False,
                "error_type": "StatisticalAssumptionError",
                "traceback": str(e),
            }

    # --- Step 3: extract and validate results_dict ------------------------
    results = isolated_ns.get("results_dict")
    if results is None:
        return {
            "success": False,
            "error_type": "OutputConventionError",
            "traceback": "code ran but results_dict was not defined",
        }

    ok, reason = validate_execution_results(results)
    if not ok:
        return {
            "success": False,
            "error_type": "ResultValidationError",
            "traceback": reason,
        }

    return {
        "success": True,
        "results": results,
        "raw_output": stdout_capture.getvalue(),
    }
