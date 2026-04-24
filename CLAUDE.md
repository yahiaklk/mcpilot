# mcpilot

Context-aware MCP server advisor — recommends the right MCP servers for your project and explains why.

## Stack
- Python 3.12, managed with uv
- FastMCP for MCP server layer
- DuckDB for local storage (data/mcp_servers.db)
- sentence-transformers (all-MiniLM-L6-v2) for semantic search
- No cloud infra — local only

## Project structure
- `src/mcpilot/server.py` — FastMCP server, three tools: recommend_for_project, recommend_next, explain_why
- `src/mcpilot/indexer.py` — parses awesome-mcp-servers README, embeds, stores in DuckDB
- `src/mcpilot/search.py` — semantic search + template-based rationale generation
- `data/mcp_servers.db` — DuckDB index (gitignored, rebuilt on first run)
- `mcpilot_brief.md` — product brief and build plan

## Running
```bash
uv sync
uv run python -m mcpilot.indexer        # build index
uv run python -m mcpilot.server         # run MCP server
uv run python -m mcpilot.indexer --force # rebuild index
```

## Status
v0.1. Three core tools implemented. Parser extracts ~2000 servers from awesome-mcp-servers. Similarity computed in DuckDB via `array_cosine_similarity`. 38 regression tests in `tests/`.
