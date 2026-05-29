"""
Runtime context that holds non-serializable objects (the dataframe) outside the
LangGraph state.

LangGraph state flows through a checkpointer and is best kept JSON-friendly. The
pandas DataFrame the Executor runs against is large and not meaningful to the
LLM, so we keep it here and have the runner set it once before invoking the graph.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

_DATAFRAME: Optional[pd.DataFrame] = None


def set_dataframe(df: pd.DataFrame) -> None:
    """Register the dataframe the Executor node will run analysis code against."""
    global _DATAFRAME
    _DATAFRAME = df


def get_dataframe() -> pd.DataFrame:
    """Return the registered dataframe; raise if the runner forgot to set it."""
    if _DATAFRAME is None:
        raise RuntimeError(
            "No dataframe registered. Call runtime.set_dataframe(df) before "
            "invoking the graph."
        )
    return _DATAFRAME
