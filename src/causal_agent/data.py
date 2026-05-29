"""
Synthetic micro-dataset with a known causal effect.

The data-generating process (DGP) embeds selection bias (treatment depends on a
confounder) and a known true ATE so the system's estimate can be quantitatively
validated against a god's-eye ground truth.

DGP
---
    historical_activity X ~ Beta(2, 5)                       (confounder, right-skewed)
    P(coupon | X)        = clip(0.2 + 0.6 * X, 0.05, 0.95)   (selection bias)
    P(click | T, X)      = clip(0.1 + tau*T + 0.3*X + eps, 0.01, 0.99)
    actual_click         ~ Binomial(1, P(click | T, X))      (binary outcome)

True ATE = tau = 0.05 (a 5 percentage-point lift in click rate).
Acceptance: estimated ATE in [0.04, 0.06] and p-value < 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SYNTH_N, SYNTH_SEED, TRUE_ATE


def generate_synthetic_data(
    n: int = SYNTH_N,
    seed: int = SYNTH_SEED,
    true_ate: float = TRUE_ATE,
) -> pd.DataFrame:
    """Generate a binary-outcome synthetic dataset with a known causal effect."""
    rng = np.random.default_rng(seed)

    # Confounder: historical activity (right-skewed in [0, 1]).
    historical_activity = rng.beta(2, 5, n)

    # Coupon assignment mechanism (selection bias: more active users are more
    # likely to receive a coupon).
    propensity = np.clip(0.2 + 0.6 * historical_activity, 0.05, 0.95)
    coupon_assigned = rng.binomial(1, propensity, n)

    # Click probability driven jointly by treatment and the confounder.
    noise = rng.normal(0, 0.03, n)
    click_prob = np.clip(
        0.1 + true_ate * coupon_assigned + 0.3 * historical_activity + noise,
        0.01,
        0.99,
    )

    # Binary click event (the v2 outcome).
    actual_click = rng.binomial(1, click_prob, n)

    df = pd.DataFrame(
        {
            "user_id": range(n),
            "coupon_assigned": coupon_assigned,         # Treatment (binary)
            "historical_activity": historical_activity,  # Confounder (continuous)
            "ctr_click": actual_click,                   # Outcome (binary)
        }
    )

    print(
        f"[data summary] n={n} | coupon rate={coupon_assigned.mean():.2%} "
        f"| overall click rate={actual_click.mean():.2%}"
    )
    print(
        f"[god's-eye view] true ATE = {true_ate} (5pp click-rate lift) "
        f"-- system target: recover this value"
    )
    return df


def build_metadata(df: pd.DataFrame) -> dict:
    """Build the metadata dict passed to the Planning Node (excludes user_id)."""
    return {
        col: {
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "sample_range": [float(df[col].min()), float(df[col].max())],
        }
        for col in df.columns
        if col != "user_id"
    }
