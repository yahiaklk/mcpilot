"""
mcpilot — context-aware MCP server advisor.

Three tools:
  recommend_for_project  — initial stack recommendation for a new project
  recommend_next         — mid-project: what to add as needs evolve
  explain_why            — why a specific server fits your project

Run with: uv run python -m mcpilot.server
"""

import sys
import traceback

from fastmcp import FastMCP

from mcpilot.indexer import build_index, is_index_ready
from mcpilot.search import find_similar, generate_rationale, lookup_by_name

mcp = FastMCP("mcpilot")

_index_initialized = False


def _ensure_index() -> None:
    global _index_initialized
    if _index_initialized:
        return
    if not is_index_ready():
        print("Index not found — building now (first run, ~30s)...", file=sys.stderr)
        build_index()
    _index_initialized = True


def _error_response(context: str, exc: Exception) -> str:
    traceback.print_exc(file=sys.stderr)
    return (
        f"## Error while {context}\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"See server logs for details. If this persists, try rebuilding the "
        f"index with `uv run python -m mcpilot.indexer --force`."
    )


def _format_results(results: list[dict], project_description: str) -> str:
    if not results:
        return "No matching MCP servers found. Try a more descriptive project description."

    lines = []
    for i, r in enumerate(results, 1):
        rationale = generate_rationale(r, project_description)
        lines.append(
            f"{i}. **{r['name']}**\n"
            f"   {r['url']}\n"
            f"   {rationale}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def recommend_for_project(description: str) -> str:
    """
    Given a project description, recommend the top MCP servers to install and explain why each one fits.
    Example: "Python FastAPI backend with PostgreSQL and JWT auth"
    """
    try:
        _ensure_index()
        results = find_similar(description, top_k=5)
        header = f"## Recommended MCP servers for: {description}\n\n"
        return header + _format_results(results, description)
    except Exception as e:
        return _error_response("generating recommendations", e)


@mcp.tool()
def recommend_next(current_stack: str, new_context: str) -> str:
    """
    Mid-project advisor: given your current MCP stack (comma-separated server names) and a new
    development context, recommend what to add next and why.
    Example: current_stack="github,filesystem", new_context="adding Stripe payments and PDF invoices"
    """
    try:
        _ensure_index()
        installed = [s.strip() for s in current_stack.split(",") if s.strip()]
        results = find_similar(new_context, top_k=5, exclude=installed)

        header = (
            f"## What to add next\n"
            f"**Current stack:** {current_stack}\n"
            f"**New context:** {new_context}\n\n"
        )
        return header + _format_results(results, new_context)
    except Exception as e:
        return _error_response("computing next recommendations", e)


@mcp.tool()
def explain_why(server_name: str, project_description: str) -> str:
    """
    Explain why a specific MCP server is a good fit for a given project.
    Example: server_name="github", project_description="open source Python library with CI/CD"
    """
    try:
        _ensure_index()
        server = lookup_by_name(server_name)
        if server is None:
            return (
                f"Could not find '{server_name}' in the index. "
                f"Try a partial name or check spelling."
            )

        rationale = generate_rationale(server, project_description)
        return (
            f"## Why {server['name']} fits your project\n\n"
            f"**Server:** {server['name']}\n"
            f"**URL:** {server['url']}\n"
            f"**Category:** {server['category']}\n"
            f"**Description:** {server['description']}\n\n"
            f"**Rationale:** {rationale}"
        )
    except Exception as e:
        return _error_response(f"explaining fit for '{server_name}'", e)


if __name__ == "__main__":
    mcp.run()
