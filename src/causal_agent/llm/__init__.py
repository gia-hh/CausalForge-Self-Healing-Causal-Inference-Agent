"""LLM backend layer: prompts, mock client, real client, dispatcher."""

from .client import call_heavy, call_light

__all__ = ["call_heavy", "call_light"]
