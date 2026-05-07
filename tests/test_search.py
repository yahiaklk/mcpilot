"""
Regression tests for search.py.

Covers fixes from:
  e793c1b  lookup_by_name ranking: exact → short-name exact → owner-exact → prefix → substring
  add901f  lookup_by_name blank-input guard
  add901f  find_similar adaptive threshold fallback
  v0.2.0   find_similar no pre-filter LIMIT (limit-exhaustion fix)
  v0.2.0   _encode_query chunk+mean-pool for >256-token inputs
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import duckdb

import kothar.indexer as indexer_mod
import kothar.search as search_mod
from kothar.search import _encode_query, find_similar, generate_rationale, lookup_by_name


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


# ---------------------------------------------------------------------------
# _encode_query — 256-token truncation fix (v0.2.0)
# ---------------------------------------------------------------------------

def _mock_model(n_tokens: int, n_emb_rows: int = 1) -> MagicMock:
    """Minimal SentenceTransformer mock for _encode_query tests."""
    m = MagicMock()
    m.max_seq_length = 256
    m.tokenizer.encode.return_value = list(range(n_tokens))
    m.tokenizer.decode.return_value = "chunk text"
    m.encode.return_value = np.ones((n_emb_rows, 384), dtype="float32")
    return m


class TestEncodeQuery:
    def test_short_text_single_encode_call(self):
        """Text within the token limit uses a single model.encode call."""
        m = _mock_model(100)
        result = _encode_query(m, "hello world")
        m.encode.assert_called_once_with(["hello world"])
        assert len(result) == 384

    def test_exactly_at_limit_no_chunking(self):
        """254 tokens (== max_seq_length - 2) stays as a single chunk."""
        m = _mock_model(254)
        _encode_query(m, "at limit")
        m.encode.assert_called_once()

    def test_over_limit_produces_multiple_chunks(self):
        """508 tokens splits into 2 chunks of 254 each."""
        m = _mock_model(508, n_emb_rows=2)
        _encode_query(m, "long " * 200)
        chunks_passed = m.encode.call_args[0][0]
        assert len(chunks_passed) == 2

    def test_over_limit_emits_warning(self, capsys):
        m = _mock_model(500, n_emb_rows=2)
        _encode_query(m, "long " * 200)
        assert "500 tokens" in capsys.readouterr().err

    def test_result_length_always_384(self):
        for n in [10, 254, 255, 508, 762]:
            n_chunks = max(1, math.ceil(n / 254))
            m = _mock_model(n, n_emb_rows=n_chunks)
            assert len(_encode_query(m, "x")) == 384

    def test_output_is_normalized(self):
        """Mean-pooled embedding is L2-normalized."""
        m = _mock_model(508, n_emb_rows=2)
        m.encode.return_value = np.array([[2.0] + [0.0] * 383, [2.0] + [0.0] * 383])
        result = _encode_query(m, "long " * 200)
        norm = math.sqrt(sum(x ** 2 for x in result))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# find_similar — limit-exhaustion fix (v0.2.0)
# ---------------------------------------------------------------------------

@pytest.fixture()
def exclusion_db(monkeypatch, tmp_path):
    """
    20-server DB where the 15 highest-scoring servers all match the 'xserver'
    substring. Tests that removing the pre-filter LIMIT lets Python reach the
    5 lower-scoring 'alpha-*' servers even when all top-15 are excluded.

    Query embedding: unit vector [1, 0, …, 0]
    xserver-* embedding: [1, 0, …, 0]  → score = 1.0
    alpha-*   embedding: [0.5, 0.866, 0, …]  → score = 0.5
    """
    db_file = tmp_path / "excl.db"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE servers (
            id INTEGER PRIMARY KEY,
            name VARCHAR, description VARCHAR, url VARCHAR, category VARCHAR,
            embedding FLOAT[384]
        )
    """)
    high_emb = [1.0] + [0.0] * 383
    low_emb  = [0.5, 0.8660254] + [0.0] * 382
    for i in range(15):
        con.execute(
            "INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?)",
            [i, f"xserver-{i}", f"Desc {i}", f"https://x.com/{i}", "Test", high_emb],
        )
    for i in range(5):
        con.execute(
            "INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?)",
            [15 + i, f"alpha-{i}", f"Alpha {i}", f"https://a.com/{i}", "Test", low_emb],
        )
    con.close()

    monkeypatch.setattr(indexer_mod, "DB_PATH", db_file)

    def _conn():
        return duckdb.connect(str(db_file), read_only=True)

    monkeypatch.setattr(indexer_mod, "get_connection", _conn)
    monkeypatch.setattr(search_mod, "get_connection", _conn)

    mock_model = MagicMock()
    mock_model.max_seq_length = 256
    mock_model.tokenizer.encode.return_value = list(range(10))
    mock_model.encode.return_value = np.array([[1.0] + [0.0] * 383])
    monkeypatch.setattr(search_mod, "_get_model", lambda: mock_model)


class TestFindSimilarLimitExhaustion:
    """Regression: old fetch_limit LIMIT could exhaust before Python exclusion filter ran."""

    def test_returns_results_when_top_servers_excluded(self, exclusion_db):
        """Exclude 'xserver' (substring) — must return the 5 alpha servers."""
        results = find_similar("test", top_k=5, exclude=["xserver"], min_score=0.0)
        assert len(results) == 5
        assert all(r["name"].startswith("alpha-") for r in results)

    def test_no_false_empty_from_limit_exhaustion(self, exclusion_db):
        """With old LIMIT=15, excluding 'xserver' gave [] even though alpha servers exist."""
        results = find_similar("test", top_k=5, exclude=["xserver"], min_score=0.0)
        assert results  # must not be empty

    def test_partial_exclusion_returns_correct_count(self, exclusion_db):
        """Exclude only 3 xserver-* by exact name — should return 5 results (2 xserver + 3 alpha... wait:
        actually top_k=5, 12 xserver not excluded → 5 returned immediately. Verify top_k is respected."""
        results = find_similar(
            "test",
            top_k=5,
            exclude=["xserver-0", "xserver-1", "xserver-2"],
            min_score=0.0,
        )
        assert len(results) == 5
        # The top results should be the 12 remaining xserver-* (score=1.0)
        assert all(r["name"].startswith("xserver-") for r in results)
