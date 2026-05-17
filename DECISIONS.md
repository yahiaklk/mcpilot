# Architectural Decision Records

## ADR-1: awesome-mcp-servers README is pinned by commit SHA, updated manually on a schedule

**Status:** Accepted

**Context:**
kothar indexes the awesome-mcp-servers README at a pinned SHA for reproducibility. The pin goes stale as new servers are added upstream.

**Decision:**
Manual update policy — rebuild the index with `uv run python -m kothar.indexer --force` when the catalog feels stale. Target: monthly or when a new server category appears. Store the current SHA and date in a comment in `indexer.py` so staleness is visible.

**Consequences:**
- No auto-update mechanism needed.
- The SHA comment in indexer.py is the staleness indicator — if the comment date is old, the index is likely stale.
- To update: fetch the current HEAD SHA from punkpeye/awesome-mcp-servers, update README_SHA in indexer.py, update the pinned date comment, and run `--force`.
