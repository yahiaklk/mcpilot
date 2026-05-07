"""
Regression tests for indexer.py — parse_readme, _extract_description, is_index_ready,
_load_local_registry.

Covers fixes from:
  a68237b  _SECTION_RE restricted to ### only
  dd51e6c  strip HTML anchors from section headings, handle Glama badges
  v0.2.0   is_index_ready opens read-only connection (duckdb-lock fix)
  local-registry  MCPILOT_LOCAL_REGISTRY env var
"""

import os

import pytest
import duckdb

import kothar.indexer as indexer_mod

from kothar.indexer import _extract_description, _load_local_registry, parse_readme


# ---------------------------------------------------------------------------
# _extract_description
# ---------------------------------------------------------------------------

class TestExtractDescription:
    def test_simple_dash_separator(self):
        assert _extract_description(" - A filesystem MCP server") == "A filesystem MCP server"

    def test_em_dash_separator(self):
        assert _extract_description(" — Provides GitHub integration") == "Provides GitHub integration"

    def test_en_dash_separator(self):
        assert _extract_description(" – Stripe payments helper") == "Stripe payments helper"

    def test_no_separator(self):
        assert _extract_description(" Search and retrieve documents") == "Search and retrieve documents"

    def test_bare_image_badge_stripped(self):
        tail = " ![Python](badge.png) - Runs Python scripts"
        assert _extract_description(tail) == "Runs Python scripts"

    def test_glama_link_wrapping_image_stripped(self):
        # [![alt](img)](link) pattern — Glama badge
        tail = " [![Glama](https://glama.ai/badge.png)](https://glama.ai/x) - Great tool"
        assert _extract_description(tail) == "Great tool"

    def test_bare_link_stripped(self):
        tail = " [MIT](https://example.com) - Does something"
        assert _extract_description(tail) == "Does something"

    def test_multiple_leading_badges_stripped(self):
        tail = " [![a](a.png)](a.com) ![b](b.png) - Final description"
        assert _extract_description(tail) == "Final description"

    def test_unicode_emoji_before_description(self):
        tail = " 🐍 - Python integration server"
        assert _extract_description(tail) == "Python integration server"

    def test_empty_tail_returns_empty(self):
        assert _extract_description("") == ""

    def test_no_ascii_letter_returns_empty(self):
        assert _extract_description(" — — 🔥 ") == ""

    def test_inline_link_preserved(self):
        # Badges only stripped from leading position; inline links stay
        result = _extract_description(" - Wraps the [Notion API](https://notion.so)")
        assert result == "Wraps the [Notion API](https://notion.so)"


# ---------------------------------------------------------------------------
# parse_readme — _SECTION_RE regression (### only)
# ---------------------------------------------------------------------------

class TestParseReadme:
    def _readme(self, *lines: str) -> str:
        return "\n".join(lines)

    def test_h3_section_sets_category(self):
        text = self._readme(
            "### Developer Tools",
            "- [mytool](https://example.com) - Does dev stuff",
        )
        servers = parse_readme(text)
        assert len(servers) == 1
        assert servers[0]["category"] == "Developer Tools"

    def test_h2_does_not_set_category(self):
        """## headings must NOT be treated as categories (regression: a68237b)."""
        text = self._readme(
            "## Should Not Be Category",
            "### Real Category",
            "- [tool](https://example.com) - A good tool",
        )
        servers = parse_readme(text)
        assert servers[0]["category"] == "Real Category"

    def test_h4_does_not_set_category(self):
        """#### headings must not be treated as categories."""
        text = self._readme(
            "#### SubSub",
            "### Actual Category",
            "- [tool](https://example.com) - A tool",
        )
        servers = parse_readme(text)
        assert servers[0]["category"] == "Actual Category"

    def test_h3_no_space_does_not_set_category(self):
        """###nospace must not match (regex requires a space after ###)."""
        text = self._readme(
            "###NoSpace",
            "### Valid Category",
            "- [tool](https://example.com) - A tool",
        )
        servers = parse_readme(text)
        assert servers[0]["category"] == "Valid Category"

    def test_html_anchor_stripped_from_category(self):
        """HTML anchors like <a name='x'></a> before heading text must be stripped."""
        text = self._readme(
            "### <a name='dev'></a>Developer Tools",
            "- [tool](https://example.com) - A tool",
        )
        servers = parse_readme(text)
        assert servers[0]["category"] == "Developer Tools"

    def test_no_description_skipped(self):
        """Bullets with no recoverable description are dropped."""
        text = self._readme(
            "### Cat",
            "- [emptytool](https://example.com)",
            "- [realtool](https://example.com) - Has description",
        )
        servers = parse_readme(text)
        assert len(servers) == 1
        assert servers[0]["name"] == "realtool"

    def test_server_fields(self):
        text = self._readme(
            "### Databases",
            "- [pgmcp](https://github.com/example/pgmcp) - PostgreSQL integration",
        )
        s = parse_readme(text)[0]
        assert s["name"] == "pgmcp"
        assert s["url"] == "https://github.com/example/pgmcp"
        assert s["description"] == "PostgreSQL integration"
        assert s["category"] == "Databases"

    def test_default_category_uncategorized(self):
        text = "- [tool](https://example.com) - A tool without a section"
        servers = parse_readme(text)
        assert servers[0]["category"] == "Uncategorized"

    def test_multiple_sections(self):
        text = self._readme(
            "### Section A",
            "- [a](https://a.com) - Tool A",
            "### Section B",
            "- [b](https://b.com) - Tool B",
        )
        servers = parse_readme(text)
        assert len(servers) == 2
        assert servers[0]["category"] == "Section A"
        assert servers[1]["category"] == "Section B"

    def test_glama_badge_in_bullet_stripped(self):
        text = self._readme(
            "### Tools",
            "- [badgetool](https://example.com) [![Glama](https://glama.ai/b.png)](https://glama.ai/x) - Actual description",
        )
        servers = parse_readme(text)
        assert servers[0]["description"] == "Actual description"


# ---------------------------------------------------------------------------
# is_index_ready — read-only connection fix (v0.2.0)
# ---------------------------------------------------------------------------

def _seed_db(path, count: int = 1):
    """Create a DB file with `count` dummy server rows."""
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE servers (
            id INTEGER PRIMARY KEY,
            name VARCHAR, description VARCHAR, url VARCHAR, category VARCHAR,
            embedding FLOAT[384]
        )
    """)
    zero = [0.0] * 384
    for i in range(count):
        con.execute(
            "INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?)",
            [i, f"s{i}", f"desc{i}", f"https://x.com/{i}", "Cat", zero],
        )
    con.close()


class TestIsIndexReady:
    def test_returns_true_when_db_has_data(self, monkeypatch, tmp_path):
        db = tmp_path / "test.db"
        _seed_db(db)
        monkeypatch.setattr(indexer_mod, "DB_PATH", db)
        assert indexer_mod.is_index_ready() is True

    def test_returns_false_when_db_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(indexer_mod, "DB_PATH", tmp_path / "nonexistent.db")
        assert indexer_mod.is_index_ready() is False

    def test_returns_false_when_table_empty(self, monkeypatch, tmp_path):
        db = tmp_path / "empty.db"
        _seed_db(db, count=0)
        monkeypatch.setattr(indexer_mod, "DB_PATH", db)
        assert indexer_mod.is_index_ready() is False

    def test_concurrent_read_only_connections_allowed(self, monkeypatch, tmp_path):
        """is_index_ready must use read_only=True so it doesn't block other readers."""
        db = tmp_path / "test.db"
        _seed_db(db)
        monkeypatch.setattr(indexer_mod, "DB_PATH", db)

        # Another process may already hold a read-only connection; is_index_ready
        # must not fail — two read-only connections to the same file are allowed.
        concurrent_ro = duckdb.connect(str(db), read_only=True)
        try:
            assert indexer_mod.is_index_ready() is True
        finally:
            concurrent_ro.close()


# ---------------------------------------------------------------------------
# _load_local_registry — MCPILOT_LOCAL_REGISTRY env var
# ---------------------------------------------------------------------------

class TestLoadLocalRegistry:
    def _write_yaml(self, path, content: str):
        path.write_text(content)
        return str(path)

    def test_env_unset_returns_empty(self, monkeypatch):
        monkeypatch.delenv("MCPILOT_LOCAL_REGISTRY", raising=False)
        assert _load_local_registry() == []

    def test_env_set_to_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", str(tmp_path / "nonexistent.yaml"))
        assert _load_local_registry() == []

    def test_valid_yaml_returns_servers(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", """
servers:
  - name: my-private-mcp
    description: Internal tool for X
    url: https://internal.example.com
    category: Internal
""")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        result = _load_local_registry()
        assert len(result) == 1
        assert result[0]["name"] == "my-private-mcp"
        assert result[0]["description"] == "Internal tool for X"
        assert result[0]["url"] == "https://internal.example.com"
        assert result[0]["category"] == "Internal"

    def test_missing_url_defaults_to_empty_string(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", """
servers:
  - name: no-url-server
    description: A server with no URL
""")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        result = _load_local_registry()
        assert result[0]["url"] == ""

    def test_missing_category_defaults_to_local(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", """
servers:
  - name: s
    description: d
""")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        assert _load_local_registry()[0]["category"] == "Local"

    def test_entry_missing_name_skipped(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", """
servers:
  - description: no name here
  - name: valid
    description: valid entry
""")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        result = _load_local_registry()
        assert len(result) == 1
        assert result[0]["name"] == "valid"

    def test_entry_missing_description_skipped(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", """
servers:
  - name: no-desc
  - name: valid
    description: has a description
""")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        result = _load_local_registry()
        assert len(result) == 1
        assert result[0]["name"] == "valid"

    def test_malformed_yaml_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{{{invalid yaml")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", str(p))
        assert _load_local_registry() == []

    def test_non_dict_root_returns_empty(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", "- just a list")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        assert _load_local_registry() == []

    def test_multiple_servers_all_returned(self, monkeypatch, tmp_path):
        p = self._write_yaml(tmp_path / "reg.yaml", """
servers:
  - name: alpha
    description: Alpha server
  - name: beta
    description: Beta server
  - name: gamma
    description: Gamma server
""")
        monkeypatch.setenv("MCPILOT_LOCAL_REGISTRY", p)
        result = _load_local_registry()
        assert len(result) == 3
        assert [r["name"] for r in result] == ["alpha", "beta", "gamma"]
