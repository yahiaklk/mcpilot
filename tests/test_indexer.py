"""
Regression tests for indexer.py — parse_readme and _extract_description.

Covers fixes from:
  a68237b  _SECTION_RE restricted to ### only
  dd51e6c  strip HTML anchors from section headings, handle Glama badges
"""

import pytest

from mcpilot.indexer import _extract_description, parse_readme


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
