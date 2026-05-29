"""
LangGraph assembly.

Wires the nodes into the state machine described in the framework: a linear spine
(planning -> codegen -> executor -> hte -> sanity -> report) with a self-repair
sub-loop reached via conditional edges, and human-escalation terminals.

A MemorySaver checkpointer is required so that ``interrupt`` can persist the
paused state and later resume via ``Command(resume=...)``.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    codegen_node,
    execution_router,
    executor_node,
    hte_supervisor_node,
    human_escalation_node,
    llm_parser_node,
    planning_node,
    repair_node,
    report_node,
    rule_based_parser_node,
    sanity_check_node,
    sanity_router,
)
from .state import AgentState


def build_graph():
    """Build and compile the causal-inference workflow graph."""
    workflow = StateGraph(AgentState)

    # Register nodes.
    workflow.add_node("planning", planning_node)
    workflow.add_node("codegen", codegen_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("rule_parser", rule_based_parser_node)
    workflow.add_node("llm_parser", llm_parser_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("hte_supervisor", hte_supervisor_node)
    workflow.add_node("sanity_check", sanity_check_node)
    workflow.add_node("report", report_node)
    workflow.add_node("human_escalation", human_escalation_node)

    # Spine edges.
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "codegen")
    workflow.add_edge("codegen", "executor")
    workflow.add_edge("rule_parser", "repair")
    workflow.add_edge("llm_parser", "repair")
    workflow.add_edge("repair", "executor")
    workflow.add_edge("hte_supervisor", "sanity_check")

    # Conditional edge: Executor success/failure routing.
    workflow.add_conditional_edges(
        "executor",
        execution_router,
        {
            "hte_supervisor": "hte_supervisor",
            "rule_based_parser": "rule_parser",
            "llm_parser": "llm_parser",
            "human_escalation": "human_escalation",
        },
    )

    # Conditional edge: Sanity check pass/fail routing.
    workflow.add_conditional_edges(
        "sanity_check",
        sanity_router,
        {
            "report": "report",
            "human_escalation": "human_escalation",
        },
    )

    workflow.add_edge("report", END)
    workflow.add_edge("human_escalation", END)

    # interrupt requires a checkpointer.
    return workflow.compile(checkpointer=MemorySaver())
