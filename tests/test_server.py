"""
Tests for server.py tool layer.

Covers:
  session-file-param  recommend_next reads session_file and appends to context
  recommend-for-goal  recommend_for_goal decomposes goal into sub-queries
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import duckdb

import kothar.indexer as indexer_mod
import kothar.search as search_mod
from kothar.server import recommend_next, recommend_for_goal


# ---------------------------------------------------------------------------
# Shared DB fixture
# ---------------------------------------------------------------------------

_SERVERS = [
    (0, "github/github-mcp", "GitHub integration", "https://github.com/github/github-mcp", "Dev Tools"),
    (1, "stripe/stripe-mcp", "Stripe payments", "https://github.com/stripe/stripe-mcp", "Finance"),
    (2, "postgres-mcp", "PostgreSQL integration", "https://github.com/x/postgres-mcp", "Databases"),
]


@pytest.fixture(autouse=True)
def patch_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE servers (
            id INTEGER PRIMARY KEY,
            name VARCHAR, description VARCHAR, url VARCHAR, category VARCHAR,
            embedding FLOAT[384]
        )
    """)
    zero = [0.0] * 384
    for row in _SERVERS:
        con.executemany("INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?)", [(*row, zero)])
    con.close()

    monkeypatch.setattr(indexer_mod, "DB_PATH", db_file)

    def _fake_connection():
        return duckdb.connect(str(db_file), read_only=True)

    monkeypatch.setattr(indexer_mod, "get_connection", _fake_connection)
    monkeypatch.setattr(search_mod, "get_connection", _fake_connection)

    mock_model = MagicMock()
    mock_model.max_seq_length = 256
    mock_model.tokenizer.encode.return_value = list(range(10))
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")
    monkeypatch.setattr(search_mod, "_get_model", lambda: mock_model)

    # Mark index as ready so _ensure_index() skips build_index
    monkeypatch.setattr(indexer_mod, "is_index_ready", lambda: True)
    import kothar.server as server_mod
    monkeypatch.setattr(server_mod, "_index_initialized", True)


# ---------------------------------------------------------------------------
# recommend_next — session_file parameter
# ---------------------------------------------------------------------------

class TestRecommendNextSessionFile:
    def test_no_session_file_works_normally(self):
        result = recommend_next(["github/github-mcp"], "add payments")
        assert "## What to add next" in result

    def test_session_file_content_appended(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        sf = tmp_path / "session.md"
        sf.write_text("Add Redis caching layer")

        captured_context = []
        original_find = search_mod.find_similar

        def _spy_find(query, **kwargs):
            captured_context.append(query)
            return original_find(query, **kwargs)

        with patch.object(search_mod, "find_similar", side_effect=_spy_find):
            recommend_next([], "payments integration", session_file=str(sf))

        assert len(captured_context) == 1
        assert "payments integration" in captured_context[0]
        assert "Add Redis caching layer" in captured_context[0]

    def test_session_file_empty_does_not_append(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        sf = tmp_path / "empty.md"
        sf.write_text("   \n")

        captured_context = []
        original_find = search_mod.find_similar

        def _spy_find(query, **kwargs):
            captured_context.append(query)
            return original_find(query, **kwargs)

        with patch.object(search_mod, "find_similar", side_effect=_spy_find):
            recommend_next([], "payments integration", session_file=str(sf))

        assert captured_context[0] == "payments integration"

    def test_missing_session_file_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        result = recommend_next([], "context", session_file=str(tmp_path / "nonexistent.md"))
        assert "Error reading session file" in result
        assert "nonexistent" in result

    def test_unreadable_session_file_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        sf = tmp_path / "locked.md"
        sf.write_text("content")
        sf.chmod(0o000)
        try:
            result = recommend_next([], "context", session_file=str(sf))
            assert "Error reading session file" in result
        finally:
            sf.chmod(0o644)

    def test_session_file_none_is_default(self):
        result = recommend_next([], "context", session_file=None)
        assert "## What to add next" in result

    def test_new_context_shown_in_header_not_session_content(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        sf = tmp_path / "session.md"
        sf.write_text("secret session notes")
        result = recommend_next([], "original context", session_file=str(sf))
        assert "original context" in result
        assert "secret session notes" not in result.split("## What to add next")[1].split("**New context:**")[1].split("\n")[0]

    def test_session_file_outside_vault_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        outside = tmp_path.parent / "outside.md"
        result = recommend_next([], "context", session_file=str(outside))
        assert "## Error" in result
        assert "session_file must be under" in result


# ---------------------------------------------------------------------------
# recommend_for_goal — decomposition and merging
# ---------------------------------------------------------------------------

class TestRecommendForGoal:
    def test_single_goal_returns_results(self):
        result = recommend_for_goal("add GitHub integration")
        assert result  # not empty

    def test_goal_with_period_split(self):
        result = recommend_for_goal("add GitHub integration. add Stripe payments")
        assert "GitHub integration" in result
        assert "Stripe payments" in result

    def test_goal_with_semicolon_split(self):
        result = recommend_for_goal("add GitHub; add Stripe")
        assert "GitHub" in result
        assert "Stripe" in result

    def test_goal_with_then_split(self):
        result = recommend_for_goal("add GitHub then add Stripe")
        assert "GitHub" in result
        assert "Stripe" in result

    def test_goal_with_comma_then_split(self):
        result = recommend_for_goal("add GitHub, then add Stripe")
        assert "GitHub" in result
        assert "Stripe" in result

    def test_bare_and_not_split(self):
        # " and " alone should NOT split into sub-queries
        result = recommend_for_goal("GitHub and Stripe")
        # Should produce a single section, not two
        assert result.count("###") <= 1

    def test_project_context_prepended(self):
        captured_queries = []
        original_find = search_mod.find_similar

        def _spy_find(query, **kwargs):
            captured_queries.append(query)
            return original_find(query, **kwargs)

        with patch.object(search_mod, "find_similar", side_effect=_spy_find):
            recommend_for_goal("add GitHub integration", project="Python FastAPI backend")

        assert all("Python FastAPI backend" in q for q in captured_queries)

    def test_deduplication_across_subgoals(self):
        # Two similar sub-queries shouldn't produce duplicate server entries
        result = recommend_for_goal("add GitHub integration. add GitHub integration")
        # Count occurrences of github-mcp; should appear at most once
        assert result.count("github/github-mcp") <= 1

    def test_output_grouped_by_subgoal(self):
        result = recommend_for_goal("add GitHub integration. add Stripe payments")
        # Should have section headers for each sub-goal
        assert "###" in result

    def test_empty_goal_returns_error_or_empty(self):
        result = recommend_for_goal("")
        # Should not crash; returns empty or an informative message
        assert isinstance(result, str)

    def test_and_then_split(self):
        result = recommend_for_goal("add GitHub and then add Stripe")
        assert "GitHub" in result
        assert "Stripe" in result
