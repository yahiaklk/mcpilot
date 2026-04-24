# mcpilot

Context-aware MCP server advisor. Tells you what to install for your specific project — and why.

## The problem

[Glama](https://glama.ai) has 19,000+ MCP servers. You have a project. Nobody bridges the gap.

LLMs asked directly hallucinate servers that don't exist and recommend from stale training data. Directories give you search, not advice.

mcpilot fills the **selection under context** gap: not "here are 19,000 options" but "for your specific project, right now, here's what you need and why."

## The two moments nobody is serving

**Project start:** "I'm building a Python data pipeline with DuckDB and FastAPI" → what do I install right now

**Mid-project:** "I just added an auth layer / I need to handle PDF ingestion" → what do I add now that I've reached this point

The second moment is more valuable. At project start, people can Google. Mid-project they're in flow.

## Three tools

```
recommend_for_project(description)
  → top MCP servers for your stack with rationale

recommend_next(current_stack, new_context)
  → what to add as your project evolves

explain_why(server_name, project_description)
  → why a specific server fits your project
```

## Install

**Prerequisites:** [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/yahiaklk/mcpilot
cd mcpilot
uv sync
```

Build the index (first run, ~30s):

```bash
uv run python -m mcpilot.indexer
```

## Add to Claude Code

```bash
claude mcp add --scope user mcpilot -- uv run --directory /path/to/mcpilot python -m mcpilot.server
```

## Add to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mcpilot": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcpilot", "python", "-m", "mcpilot.server"]
    }
  }
}
```

## Usage

Once connected, ask your AI assistant:

```
recommend_for_project("Python FastAPI backend with PostgreSQL and JWT auth")

recommend_next("github,filesystem", "adding Stripe payments and PDF invoices")

explain_why("postgres", "multi-tenant SaaS with row-level security")
```

## How it works

- Parses [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) (2000+ curated servers)
- Embeds descriptions with `all-MiniLM-L6-v2` (local, no API cost)
- Stores in DuckDB, queries with cosine similarity
- Template-based rationale — grounded in the registry, not hallucinated

## Rebuild the index

```bash
uv run python -m mcpilot.indexer --force
```

## Stack

Python · FastMCP · DuckDB · sentence-transformers · uv

## License

[MIT](LICENSE)
