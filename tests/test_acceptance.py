"""
Acceptance tests mirroring the framework's MVP validation table.

Run with:  python -m pytest tests/ -v   (after `pip install pytest`)
or simply: python tests/test_acceptance.py   (runs a lightweight assert suite)

These tests use LLM_BACKEND=mock so they are fully offline and deterministic.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("LLM_BACKEND", "mock")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from causal_agent import config, runtime  # noqa: E402
from causal_agent.data import build_metadata, generate_synthetic_data  # noqa: E402
from causal_agent.executor import (  # noqa: E402
    StatisticalAssumptionError,
    _run_statistical_diagnostics,
)
from causal_agent.graph import build_graph  # noqa: E402

import statsmodels.api as sm  # noqa: E402


def _run_once(inject: str | None = None) -> dict:
    if inject:
        os.environ["INJECT_FAULT"] = inject
    else:
        os.environ.pop("INJECT_FAULT", None)
    df = generate_synthetic_data()
    runtime.set_dataframe(df)
    graph = build_graph()
    cfg = {"configurable": {"thread_id": f"test_{inject or 'happy'}"}}
    return graph.invoke(
        {
            "query": "test",
            "metadata": build_metadata(df),
            "repair_attempts": 0,
            "error_blacklist": [],
        },
        config=cfg,
    )


def test_ate_recovery_happy_path():
    """Estimated ATE in [0.04, 0.06] with p < 0.05 and zero repairs."""
    state = _run_once()
    r = state["execution_results"]
    assert config.ATE_LOWER_BOUND <= r["ate"] <= config.ATE_UPPER_BOUND, r["ate"]
    assert r["p_value"] < config.P_VALUE_THRESHOLD, r["p_value"]
    assert state.get("repair_attempts", 0) == 0
    assert state.get("final_report")


def test_confounding_is_corrected():
    """Naive difference is biased above the truth; IPW estimate converges to 0.05.

    Note: the framework's table cites a naive band of [0.09, 0.11], which applied
    to the OLD continuous-CTR DGP. Under the v2 binary outcome the binomial draw
    and probability clipping attenuate the gap to ~0.057, so we assert the
    direction of the bias (naive > true) rather than the legacy magnitude.
    """
    df = generate_synthetic_data()
    naive = (
        df.loc[df.coupon_assigned == 1, "ctr_click"].mean()
        - df.loc[df.coupon_assigned == 0, "ctr_click"].mean()
    )
    # Naive difference is inflated by selection bias relative to the true 0.05.
    assert naive > 0.05, f"expected biased naive diff > 0.05, got {naive:.4f}"
    state = _run_once()
    assert abs(state["execution_results"]["ate"] - 0.05) < 0.02


def test_hte_direction_correct():
    """High-activity stratum ATE exceeds low-activity stratum ATE (per the DGP)."""
    state = _run_once()
    strat = state["execution_results"]["stratified_results"]
    high = strat["historical_activity_high"]["ate"]
    low = strat["historical_activity_low"]["ate"]
    assert high > low, f"high {high} should exceed low {low}"


def test_repair_on_nameerror():
    """An injected NameError is repaired within the attempt budget."""
    state = _run_once(inject="nameerror")
    assert state.get("final_report")
    assert 1 <= state.get("repair_attempts", 0) <= config.MAX_REPAIR_ATTEMPTS


def test_repair_on_multicollinearity():
    """Injected multicollinearity trips the host VIF check and is repaired."""
    state = _run_once(inject="multicollinearity")
    assert state.get("final_report")
    assert 1 <= state.get("repair_attempts", 0) <= config.MAX_REPAIR_ATTEMPTS


def test_vif_diagnostic_fires():
    """The host VIF diagnostic raises on perfect collinearity."""
    rng = np.random.default_rng(0)
    n = 500
    T = rng.binomial(1, 0.4, n).astype(float)
    x = rng.beta(2, 5, n)
    Y = rng.binomial(1, 0.1 + 0.05 * T + 0.3 * x, n).astype(float)
    Xbad = sm.add_constant(np.column_stack([T, x, x]))  # x duplicated
    fit = sm.WLS(Y, Xbad, weights=np.ones(n)).fit()
    raised = False
    try:
        _run_statistical_diagnostics(fit)
    except StatisticalAssumptionError:
        raised = True
    assert raised, "VIF diagnostic should raise on perfect collinearity"


def test_robust_fit_passes_bp():
    """An HC3 LPM fit must NOT be flagged by the BP diagnostic."""
    rng = np.random.default_rng(1)
    n = 800
    T = rng.binomial(1, 0.4, n).astype(float)
    Y = rng.binomial(1, 0.1 + 0.05 * T, n).astype(float)
    X = sm.add_constant(T)
    fit = sm.WLS(Y, X, weights=np.ones(n)).fit(cov_type="HC3")
    # Should not raise.
    _run_statistical_diagnostics(fit)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _main()
