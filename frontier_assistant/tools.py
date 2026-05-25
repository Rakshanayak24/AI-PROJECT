"""
Tool definitions for the frontier assistant (Claude Sonnet).
Tools are passed to the Anthropic API as the `tools` parameter.
Tool results are handled in app.py.
"""

import math
import datetime
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Tool schemas (passed to Anthropic API)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Use this for arithmetic, algebra, or any calculation the user asks for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '2 ** 10' or 'math.sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Return the current date and time in ISO 8601 format (UTC).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "web_search_stub",
        "description": (
            "Stub for a web search tool. In a production deployment this would call "
            "a real search API (e.g. Brave, Serper, Tavily). Currently returns a "
            "placeholder indicating the query was received."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_tool(name: str, tool_input: Dict[str, Any]) -> str:
    """Dispatch a tool call and return the result as a string."""
    if name == "calculator":
        return _run_calculator(tool_input["expression"])
    elif name == "get_current_datetime":
        return _run_datetime()
    elif name == "web_search_stub":
        return _run_web_search(tool_input["query"])
    else:
        return f"Unknown tool: {name}"


def _run_calculator(expression: str) -> str:
    try:
        # Safe eval: only math module + builtins
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed_names["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def _run_datetime() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


def _run_web_search(query: str) -> str:
    # Stub — replace with real search API in production
    return (
        f"[Web search stub] Query received: '{query}'. "
        "In production, this would return live search results. "
        "Please note that my knowledge has a training cutoff and I cannot "
        "browse the web in this demo."
    )
