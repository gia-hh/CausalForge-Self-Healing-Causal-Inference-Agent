"""
Error-blacklist construction, repair-context formatting, and the rule-based
parser. These keep the repair loop from looping forever (Tabu-search style) and
bound the repair context so it does not grow with each attempt.
"""

from __future__ import annotations

import re


def build_blacklist_entry(parsed_error: dict) -> dict:
    """Construct one append-only blacklist entry from a parsed error."""
    return {
        "error_type": parsed_error.get("error_type"),
        "location": (
            f"line {parsed_error.get('error_line', '?')}: "
            f"{parsed_error.get('code_snippet', '')}"
        ),
        "approach_tried": parsed_error.get("approach_tried"),
    }


def format_blacklist(blacklist: list) -> str:
    """Render the blacklist for inclusion in the Repair prompt."""
    if not blacklist:
        return "(first repair; no prior blacklist)"
    lines = ["The following repair paths have been tried and failed; do NOT repeat:"]
    for i, e in enumerate(blacklist, 1):
        lines.append(
            f"  #{i}: location[{e.get('location', '?')}] "
            f"error[{e.get('error_type', '?')}] "
            f"tried method <{e.get('approach_tried', '?')}> -> failed"
        )
    lines.append("Explore an entirely new path other than all of the above.")
    return "\n".join(lines)


def build_repair_context(latest_code: str, latest_parsed_error: dict, blacklist: list) -> dict:
    """Assemble the bounded fields the Repair prompt needs.

    Returns a dict of placeholder values (not the full prompt) so the node can
    format the prompt template. Only the latest code + latest error + blacklist
    summary are included; historical code versions are intentionally discarded to
    keep the context window from growing with repair rounds.
    """
    return {
        "latest_code": latest_code,
        "error_type": latest_parsed_error.get("error_type", "UnknownError"),
        "error_line": latest_parsed_error.get("error_line", -1),
        "code_snippet": latest_parsed_error.get("code_snippet", "unknown"),
        "semantic_summary": latest_parsed_error.get("semantic_summary", "unknown"),
        "formatted_blacklist": format_blacklist(blacklist),
    }


def rule_based_parser(code: str, error_info: dict) -> dict:
    """Deterministic parser for syntactic/structural errors (no LLM)."""
    tb_str = error_info.get("traceback", "")
    lines = code.split("\n")

    line_match = re.search(r"line (\d+)", tb_str)
    error_line = int(line_match.group(1)) if line_match else -1
    if 0 < error_line <= len(lines):
        code_snippet = lines[error_line - 1].strip()
    else:
        code_snippet = "unknown"

    tb_lines = [l.strip() for l in tb_str.strip().splitlines() if l.strip()]
    key_msg = tb_lines[-1] if tb_lines else "unknown error"

    return {
        "error_type": error_info.get("error_type", "UnknownError"),
        "error_line": error_line,
        "code_snippet": code_snippet,
        "semantic_summary": key_msg,
        "approach_tried": f"original implementation near line {error_line}",
    }


def validate_and_parse_llm_output(raw: str) -> tuple[bool, dict]:
    """Validate the LLM parser's JSON output; signal fallback on failure."""
    import json

    required = {
        "error_type",
        "error_line",
        "code_snippet",
        "semantic_summary",
        "approach_tried",
    }
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return False, {}
    if not isinstance(parsed, dict) or (required - parsed.keys()):
        return False, {}
    return True, parsed
