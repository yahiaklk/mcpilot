# kothar

Context-aware capability advisor — recommends the right MCP servers, skills, and tools for your goal and explains why.

## Stack
- Python 3.12, managed with uv
- FastMCP for MCP server layer
- DuckDB for local storage (data/mcp_servers.db)
- sentence-transformers (all-MiniLM-L6-v2) for semantic search
- No cloud infra — local only

## Project structure
- `src/kothar/server.py` — FastMCP server: recommend_for_project, recommend_next, recommend_for_goal, explain_why
- `src/kothar/indexer.py` — parses awesome-mcp-servers README, embeds, stores in DuckDB
- `src/kothar/search.py` — semantic search + template-based rationale generation
- `data/mcp_servers.db` — DuckDB index (gitignored, rebuilt on first run)

## Running
```bash
uv sync
uv run python -m kothar.indexer        # build index
uv run python -m kothar.server         # run MCP server
uv run python -m kothar.indexer --force # rebuild index
```

## Tests
```bash
uv run pytest           # 79 tests
uv run ruff check src/ tests/
```

## Registration with Claude Code
```bash
claude mcp add -s user kothar -- uv run --directory ~/projects/kothar python -m kothar.server
```

## Environment variables
- `VAULT_PATH` — allowed base directory for `session_file` in `recommend_for_next_step` (default: `~/projects`)
- `KOTHAR_LOCAL_REGISTRY` — path to a YAML file of private server entries merged into the index at build time

## Status
v0.4 — 4 tools: recommend_for_project, recommend_for_next_step, recommend_for_goal, explain_fit. 80 tests passing.
