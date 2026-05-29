"""
Global shared state for the LangGraph causal-inference workflow.

Every node reads from and writes to this TypedDict. Fields annotated with a
reducer (operator.add) accumulate across super-steps; all other fields are
overwritten by the returning node.

Design notes
------------
* ``latest_execution_error`` is the RAW error emitted by the Executor (error_type,
  traceback). Routers and parsers read from this field. It is distinct from
  ``latest_parsed_error`` which holds the PARSED, structured diagnosis.
* ``repair_attempts`` uses an additive reducer. The Repair node returns
  ``{"repair_attempts": 1}`` each pass, so the running total equals the number
  of repairs performed. The router escalates once it reaches MAX_REPAIR_ATTEMPTS.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input layer ────────────────────────────────────────────────────────
    query: str
    metadata: dict

    # ── Experiment-design layer ─────────────────────────────────────────────
    experiment_design: dict
    # Example structure:
    # {
    #   "data_type": "observational",
    #   "variable_roles": {
    #       "treatment": "coupon_assigned",
    #       "outcome": "ctr_click",
    #       "outcome_type": "binary",
    #       "confounders": ["historical_activity"],
    #       "unclear_variables": []
    #   },
    #   "dag": {"nodes": [...], "edges": [...]},
    #   "method": "PSM",
    #   "method_rationale": "...",
    #   "key_assumptions": [...]
    # }

    # ── Code-execution layer ─────────────────────────────────────────────────
    latest_code: str                                    # always the newest version only
    latest_execution_error: dict                        # RAW error from Executor (overwritten)
    latest_parsed_error: dict                           # PARSED diagnosis (overwritten)
    error_blacklist: Annotated[list, operator.add]      # append-only
    repair_attempts: Annotated[int, operator.add]       # additive: +1 per repair

    # ── Results layer ────────────────────────────────────────────────────────
    execution_results: dict
    # Example structure (v2 adds stratified_results and propensity_scores):
    # {
    #   "method": "PSM-IPW",
    #   "ate": 0.048,
    #   "p_value": 0.023,
    #   "confidence_interval": [0.012, 0.084],
    #   "sample_size": 4836,
    #   "stratified_results": {
    #       "historical_activity_high": {"ate": 0.062, "p_value": 0.015, "n": 2400},
    #       "historical_activity_low":  {"ate": 0.031, "p_value": 0.041, "n": 2400}
    #   },
    #   "propensity_scores": [...]   # used by Executor diagnostics, not fed to the LLM
    # }

    hte_results: dict                                   # LLM business interpretation
    sanity_check_passed: bool
    sanity_check_details: list

    # ── Output layer ─────────────────────────────────────────────────────────
    final_report: str

    # ── Control / bookkeeping ────────────────────────────────────────────────
    halt_reason: str                                    # set when escalating / aborting
