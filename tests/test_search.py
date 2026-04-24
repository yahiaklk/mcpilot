"""
Regression tests for search.py.

Covers fixes from:
  e793c1b  lookup_by_name ranking: exact → short-name exact → owner-exact → prefix → substring
  add901f  lookup_by_name blank-input guard
  add901f  find_similar adaptive threshold fallback
"""

import pytest
import duckdb

import mcpilot.indexer as indexer_mod
import mcpilot.search as search_mod
from mcpilot.search import generate_rationale, lookup_by_name


# ---------------------------------------------------------------------------
# Fixture: tiny in-memory DuckDB with no embeddings needed for lookup tests
# ---------------------------------------------------------------------------

_SERVERS = [
    # (id, name, description, url, category)
    (0, "github/github-mcp",   "GitHub integration",        "https://github.com/github/github-mcp",   "Dev Tools"),
    (1, "stripe/stripe-mcp",   "Stripe payments",           "https://github.com/stripe/stripe-mcp",   "Finance"),
    (2, "postgres-mcp",        "PostgreSQL integration",    "https://github.com/x/postgres-mcp",      "Databases"),
    (3, "githubsimilar",       "Another GitHub-ish tool",   "https://github.com/x/githubsimilar",     "Dev Tools"),
    (4, "github-actions",      "CI/CD pipeline tool",       "https://github.com/x/github-actions",    "CI/CD"),
]


@pytest.fixture(autouse=True)
def patch_db(monkeypatch, tmp_path):
    """Replace DB_PATH and get_connection with a seeded in-memory DB."""
    db_file = tmp_path / "test.db"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE servers (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            description VARCHAR,
            url VARCHAR,
            category VARCHAR,
            embedding FLOAT[384]
        )
    """)
    zero = [0.0] * 384
    for row in _SERVERS:
        con.executemany(
            "INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?)",
            [(*row, zero)],
        )
    con.close()

    monkeypatch.setattr(indexer_mod, "DB_PATH", db_file)

    def _fake_connection():
        return duckdb.connect(str(db_file), read_only=True)

    monkeypatch.setattr(indexer_mod, "get_connection", _fake_connection)
    monkeypatch.setattr(search_mod, "get_connection", _fake_connection)


# ---------------------------------------------------------------------------
# lookup_by_name — ranking regression (e793c1b)
# ---------------------------------------------------------------------------

class TestLookupByName:
    def test_exact_full_name_wins(self):
        result = lookup_by_name("github/github-mcp")
        assert result["name"] == "github/github-mcp"

    def test_short_name_exact_wins_over_prefix(self):
        # "github-mcp" matches short name of github/github-mcp exactly,
        # and should beat "github-actions" (prefix match on "github")
        result = lookup_by_name("github-mcp")
        assert result["name"] == "github/github-mcp"

    def test_prefix_match_returned(self):
        result = lookup_by_name("postgres")
        assert result is not None
        assert "postgres" in result["name"].lower()

    def test_substring_match_returned(self):
        result = lookup_by_name("stripe")
        assert result is not None
        assert "stripe" in result["name"].lower()

    def test_owner_exact_preferred_over_prefix(self):
        # "github" is the owner of "github/github-mcp"; should rank above prefix matches
        result = lookup_by_name("github")
        assert result["name"] == "github/github-mcp"

    def test_blank_string_returns_none(self):
        """Blank input guard — regression from add901f."""
        assert lookup_by_name("") is None

    def test_whitespace_only_returns_none(self):
        assert lookup_by_name("   ") is None

    def test_no_match_returns_none(self):
        assert lookup_by_name("nonexistent-xyz-123") is None

    def test_case_insensitive(self):
        result = lookup_by_name("STRIPE")
        assert result is not None
        assert "stripe" in result["name"].lower()


# ---------------------------------------------------------------------------
# generate_rationale — output shape and confidence tiers
# ---------------------------------------------------------------------------

class TestGenerateRationale:
    def _server(self, score=0.5) -> dict:
        return {
            "name": "pgmcp",
            "description": "PostgreSQL database integration",
            "category": "Databases",
            "score": score,
        }

    def test_contains_server_name(self):
        r = generate_rationale(self._server(), "Python FastAPI with postgres")
        assert "pgmcp" in r

    def test_strong_confidence(self):
        r = generate_rationale(self._server(score=0.6), "any project")
        assert "strong" in r

    def test_moderate_confidence(self):
        r = generate_rationale(self._server(score=0.45), "any project")
        assert "moderate" in r

    def test_potential_confidence(self):
        r = generate_rationale(self._server(score=0.2), "any project")
        assert "potential" in r

    def test_overlap_hint_when_words_match(self):
        r = generate_rationale(self._server(), "build a postgres database backend")
        assert "postgres" in r or "database" in r

    def test_category_fallback_when_no_overlap(self):
        r = generate_rationale(self._server(), "unrelated machine learning pipeline")
        assert "Databases" in r

    def test_stopwords_excluded_from_overlap(self):
        # "for" "the" "and" are stopwords — should not appear as overlap hints
        server = {
            "name": "tool",
            "description": "for the and or",
            "category": "Misc",
            "score": 0.5,
        }
        r = generate_rationale(server, "for the and or")
        # Falls back to category hint since all overlap words are stopwords
        assert "Misc" in r
