"""
Pipeline runner — the single entry point to execute the whole system.

Usage
-----
    python run_pipeline.py                # happy path
    INJECT_FAULT=nameerror python run_pipeline.py        # exercise rule-parser repair
    INJECT_FAULT=multicollinearity python run_pipeline.py  # exercise stat-diagnostic repair
    LLM_BACKEND=real ANTHROPIC_API_KEY=... python run_pipeline.py  # real backends

The runner:
  1. generates the synthetic dataset and metadata,
  2. registers the dataframe in the runtime context,
  3. invokes the graph,
  4. handles any LangGraph interrupt (HITL) by prompting on the console,
  5. prints the final report or escalation reason and an acceptance summary.
"""

from __future__ import annotations

import sys

# Make the package importable when run directly from the project root.
sys.path.insert(0, "src")

from langgraph.types import Command  # noqa: E402

from causal_agent import config, runtime  # noqa: E402
from causal_agent.data import build_metadata, generate_synthetic_data  # noqa: E402
from causal_agent.graph import build_graph  # noqa: E402


DEFAULT_QUERY = (
    "We sent discount coupons to some users. Did receiving a coupon causally "
    "increase the click-through rate, and does the effect differ across user "
    "segments?"
)


def _pending_interrupt(graph, result, config_dict):
    """Return the pending interrupt payload, or None.

    Robust across LangGraph versions: newer builds put an ``__interrupt__`` key
    in the invoke result, while 0.2.x surfaces it on the state snapshot's task
    list. We check both.
    """
    if isinstance(result, dict) and "__interrupt__" in result:
        return result["__interrupt__"][0].value
    snapshot = graph.get_state(config_dict)
    for task in getattr(snapshot, "tasks", ()):
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0].value
    return None


def _resolve_interrupt(graph, payload, config_dict):
    """Prompt the user for each unclear variable, then resume the graph."""
    print(f"\n[HITL] {payload['instruction']}")
    decisions = {}
    for var in payload["unclear_variables"]:
        ans = input(f"  Is [{var}] a confounder? (y/n): ").strip().lower()
        decisions[var] = ans
    return graph.invoke(Command(resume=decisions), config=config_dict)


def run(query: str = DEFAULT_QUERY) -> dict:
    """Run the full pipeline once and return the final state."""
    print("=" * 60)
    print("Automated Causal Inference Expert System v2")
    print(f"backend: {config.describe_backend()}")
    print("=" * 60)

    # 1-2. Data + runtime registration.
    df = generate_synthetic_data()
    metadata = build_metadata(df)
    runtime.set_dataframe(df)

    # 3. Build + invoke the graph.
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "session_001"}}
    initial_input = {
        "query": query,
        "metadata": metadata,
        "repair_attempts": 0,
        "error_blacklist": [],
    }

    result = graph.invoke(initial_input, config=cfg)

    # 4. Handle HITL interrupt(s) if any.
    payload = _pending_interrupt(graph, result, cfg)
    while payload is not None:
        result = _resolve_interrupt(graph, payload, cfg)
        payload = _pending_interrupt(graph, result, cfg)

    # 5. Report / summary. The invoke result after resume holds the latest delta;
    # read the full accumulated state from the checkpointer for the summary.
    final_state = dict(graph.get_state(cfg).values)
    _print_summary(final_state)
    return final_state


def _print_summary(state: dict) -> None:
    """Print the final report (if any) and a ground-truth acceptance summary."""
    print("\n" + "=" * 60)
    if state.get("final_report"):
        print("FINAL REPORT")
        print("=" * 60)
        print(state["final_report"])
    else:
        print(f"NO REPORT — halt reason: {state.get('halt_reason', 'unknown')}")
    print("=" * 60)

    results = state.get("execution_results", {})
    ate = results.get("ate")
    p = results.get("p_value")
    if ate is not None:
        in_band = config.ATE_LOWER_BOUND <= ate <= config.ATE_UPPER_BOUND
        sig = p is not None and p < config.P_VALUE_THRESHOLD
        print("\n[ACCEPTANCE CHECK vs ground truth tau=0.05]")
        print(f"  estimated ATE = {ate:.4f}  (target [0.04, 0.06]) -> "
              f"{'PASS' if in_band else 'FAIL'}")
        print(f"  p-value       = {p:.4f}  (target < 0.05)        -> "
              f"{'PASS' if sig else 'FAIL'}")
        repairs = state.get("repair_attempts", 0)
        print(f"  repair attempts used = {repairs}")


if __name__ == "__main__":
    run()
