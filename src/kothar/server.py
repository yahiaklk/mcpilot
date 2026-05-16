"""
kothar — context-aware capability advisor.

Four tools:
  recommend_for_project  — initial stack recommendation for a new project
  recommend_for_next_step — mid-project: what to add as needs evolve
  explain_fit             — why a specific server fits your project
  recommend_for_goal     — decompose a multi-part goal into sub-queries and recommend per part

Run with: uv run python -m kothar.server
"""

import os
import re
import sys
import traceback
from pathlib import Path

from fastmcp import FastMCP

from kothar.indexer import build_index, is_index_ready
from kothar.search import find_similar, generate_rationale, lookup_by_name

mcp = FastMCP("kothar")

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
        f"index with `uv run python -m kothar.indexer --force`."
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
def recommend_for_next_step(
    current_stack: list[str],
    new_context: str,
    session_file: str | None = None,
) -> str:
    """
    Mid-project advisor: given your current MCP stack (list of server names) and a new
    development context, recommend what to add next and why.
    Example: current_stack=["github", "filesystem"], new_context="adding Stripe payments and PDF invoices"
    session_file: optional path to a session notes file whose content is appended to new_context.
    """
    try:
        _ensure_index()
        installed = [s.strip() for s in current_stack if s.strip()]

        context = new_context
        if session_file:
            allowed = Path(os.environ.get("VAULT_PATH", "~/projects")).expanduser().resolve()
            p = Path(session_file).resolve()
            if not p.is_relative_to(allowed):
                return (
                    f"## Error\n\n"
                    f"session_file must be under {allowed}"
                )
            try:
                with open(p) as f:
                    session_content = f.read().strip()
                if session_content:
                    context = f"{new_context}\n\n{session_content}"
            except OSError as e:
                return (
                    f"## Error reading session file\n\n"
                    f"Could not read `{session_file}`: {e}\n\n"
                    f"Fix the path or omit `session_file` to proceed without it."
                )

        results = find_similar(context, top_k=5, exclude=installed)

        header = (
            f"## What to add next\n"
            f"**Current stack:** {current_stack}\n"
            f"**New context:** {new_context}\n\n"
        )
        return header + _format_results(results, context)
    except Exception as e:
        return _error_response("computing next recommendations", e)


@mcp.tool()
def explain_fit(server_name: str, project_description: str) -> str:
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


# Splits on `. `, `; `, ` and then `, `, then `, ` then ` — NOT bare ` and `.
# Longer alternatives listed first so regex tries them before shorter overlaps.
_GOAL_SPLIT_RE = re.compile(
    r"\s+and\s+then\s+|,\s+then\s+|\s+then\s+|\.\s+|;\s+",
    re.IGNORECASE,
)


def _split_goal(goal: str) -> list[str]:
    return [p.strip() for p in _GOAL_SPLIT_RE.split(goal) if p.strip()]


@mcp.tool()
def recommend_for_goal(goal: str, project: str | None = None) -> str:
    """
    Decompose a multi-part goal into sub-queries and recommend MCP servers for each part.
    Splits on hard boundaries: '. ', '; ', ' then ', ', then ', ' and then ' (not bare ' and ').
    project: optional project context prepended to each sub-query for richer semantic matching.
    Example: goal="integrate GitHub. add Stripe payments", project="Python FastAPI backend"
    """
    try:
        _ensure_index()

        if not goal or not goal.strip():
            return "Please provide a goal description."

        parts = _split_goal(goal)
        seen_names: set[str] = set()

        if len(parts) == 1:
            query = f"{project}\n\n{parts[0]}" if project else parts[0]
            results = find_similar(query, top_k=5)
            header = f"## Recommended MCP servers for: {goal}\n\n"
            return header + _format_results(results, query)

        sections: list[str] = []
        for part in parts:
            query = f"{project}\n\n{part}" if project else part
            results = find_similar(query, top_k=5)
            fresh = [r for r in results if r["name"] not in seen_names]
            seen_names.update(r["name"] for r in fresh)
            sections.append(f"### {part}\n\n{_format_results(fresh, query)}")

        return "\n\n".join(sections)
    except Exception as e:
        return _error_response("computing goal recommendations", e)


def main() -> None:
    """Entry point for the `kothar` console script (see pyproject [project.scripts])."""
    mcp.run()


if __name__ == "__main__":
    main()
