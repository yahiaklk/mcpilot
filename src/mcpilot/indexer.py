"""
Parse awesome-mcp-servers README, embed descriptions, store in DuckDB.
Run directly to (re)build the index: uv run python -m mcpilot.indexer
"""

import re
import sys
from pathlib import Path

import duckdb
import requests
from sentence_transformers import SentenceTransformer

# Pinned to a specific commit for reproducible indexing. To update: fetch the
# current HEAD SHA from punkpeye/awesome-mcp-servers and update both values.
README_SHA = "a56e86528faea42a80df60e29c0ab3d1203af09f"
README_URL = f"https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/{README_SHA}/README.md"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "mcp_servers.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Server bullet: `- [name](url)` with optional leading whitespace. The remainder
# of the line (badges, emojis, separators, description) is captured as `tail`
# and handed to _extract_description for cleanup.
_BULLET_RE = re.compile(r"^\s*-\s+\[([^\]]+)\]\(([^)]+)\)(.*)$")
_SECTION_RE = re.compile(r"^###\s+(.+)$")

# Leading-position badge patterns. Matched greedy-first so a nested
# [![alt](img)](link) badge is consumed as one token rather than twice.
_LEADING_BADGES = (
    re.compile(r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)"),  # link-wrapping-image (Glama badges)
    re.compile(r"^!\[[^\]]*\]\([^)]*\)"),               # bare image
    re.compile(r"^\[[^\]]*\]\([^)]*\)"),                # bare link
)


def _extract_description(tail: str) -> str:
    """
    Recover the human-readable description from the text after `](url)`.

    Handles: varied separators (`-`, `–`, `—`, or none), language/scope emojis,
    and Glama/shields badges. Only strips badges from the leading position, so
    inline markdown links inside the description body are preserved.
    """
    s = tail
    while True:
        prev_len = len(s)
        s = s.lstrip()
        for pat in _LEADING_BADGES:
            s = pat.sub("", s, count=1)
        if len(s) == prev_len:
            break
    # The description always starts at the first ASCII letter — everything
    # preceding it is separator punctuation or unicode emojis.
    m = re.search(r"[A-Za-z]", s)
    if not m:
        return ""
    return s[m.start():].strip()


def parse_readme(text: str) -> list[dict]:
    """Return list of {name, url, description, category} from README markdown."""
    servers = []
    current_category = "Uncategorized"

    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            raw = re.sub(r"[^\w\s&/\-]", "", re.sub(r"<[^>]+>", "", m.group(1))).strip()
            if raw:
                current_category = raw
            continue

        m = _BULLET_RE.match(line)
        if not m:
            continue
        name, url, tail = m.groups()
        description = _extract_description(tail)
        if not description:
            continue
        servers.append(
            {
                "name": name.strip(),
                "url": url.strip(),
                "description": description,
                "category": current_category,
            }
        )

    return servers


def build_index(force: bool = False) -> int:
    """
    Download README, embed descriptions, store in DuckDB.
    Returns number of servers indexed.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    if not force:
        try:
            count = con.execute("SELECT COUNT(*) FROM servers").fetchone()[0]
            if count > 0:
                print(f"Index already exists ({count} servers). Use force=True to rebuild.")
                con.close()
                return count
        except duckdb.CatalogException:
            pass  # servers table doesn't exist yet

    print("Fetching awesome-mcp-servers README...")
    resp = requests.get(README_URL, timeout=30)
    resp.raise_for_status()

    servers = parse_readme(resp.text)
    print(f"Parsed {len(servers)} servers.")

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [f"{s['name']} — {s['description']}" for s in servers]
    print("Embedding descriptions (this takes ~30s on first run)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    con.execute("DROP TABLE IF EXISTS servers")
    con.execute("""
        CREATE TABLE servers (
            id      INTEGER PRIMARY KEY,
            name    VARCHAR,
            description VARCHAR,
            url     VARCHAR,
            category VARCHAR,
            embedding FLOAT[384]
        )
    """)

    rows = [
        (i, s["name"], s["description"], s["url"], s["category"], embeddings[i].tolist())
        for i, s in enumerate(servers)
    ]
    con.executemany("INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.close()

    print(f"Index built: {len(servers)} servers stored in {DB_PATH}")
    return len(servers)


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def is_index_ready() -> bool:
    if not DB_PATH.exists():
        return False
    try:
        con = duckdb.connect(str(DB_PATH))
        count = con.execute("SELECT COUNT(*) FROM servers").fetchone()[0]
        con.close()
        return count > 0
    except (duckdb.Error, OSError):
        return False


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_index(force=force)
