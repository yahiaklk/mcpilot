"""
Semantic search over the MCP server index + rationale generation.
"""

import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from kothar.indexer import MODEL_NAME, get_connection

# Similarity below this is weaker than typical near-miss matches we've measured
# with all-MiniLM-L6-v2 against the awesome-mcp-servers corpus. Tune together
# with the embedding model; not exposed on the MCP tool surface.
DEFAULT_MIN_SCORE = 0.35

STOPWORDS = frozenset({"a", "an", "the", "and", "or", "for", "to", "in", "with", "of", "that", "is", "are", "your", "you"})

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _encode_query(model: SentenceTransformer, text: str) -> list[float]:
    """
    Encode query text. Chunks + mean-pools if over the model's token limit
    so tail content isn't silently dropped by the 256-token cap.
    """
    chunk_size = model.max_seq_length - 2  # reserve 2 for [CLS] / [SEP]
    tokens = model.tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= chunk_size:
        return model.encode([text])[0].tolist()

    print(
        f"[kothar] Query is {len(tokens)} tokens (limit: {chunk_size}); "
        "chunking and mean-pooling for full coverage.",
        file=sys.stderr,
    )
    chunks = [
        model.tokenizer.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]
    embeddings = model.encode(chunks)
    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.tolist()


def find_similar(
    query: str,
    top_k: int = 10,
    exclude: list[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict]:
    """
    Return top_k servers most semantically similar to query.
    Each result: {name, url, description, category, score}
    """
    model = _get_model()
    query_emb = _encode_query(model, query)

    excluded_lower = {n.lower() for n in (exclude or [])}

    con = get_connection()
    rows = con.execute(
        """
        SELECT name, description, url, category, score FROM (
            SELECT name, description, url, category,
                   array_cosine_similarity(embedding, ?::FLOAT[384]) AS score
            FROM servers
        )
        WHERE score >= ?
        ORDER BY score DESC
        """,
        [query_emb, min_score],
    ).fetchall()
    con.close()

    results = []
    for name, desc, url, cat, score in rows:
        name_lower = name.lower()
        short_name = name_lower.split("/")[-1]
        if any(
            e == name_lower or e == short_name or e in name_lower
            for e in excluded_lower
        ):
            continue
        results.append(
            {
                "name": name,
                "url": url,
                "description": desc,
                "category": cat,
                "score": float(score),
            }
        )
        if len(results) >= top_k:
            break

    # Adaptive fallback: if nothing cleared the threshold, return the closest
    # matches below it so callers never silently get zero results.
    if not results and min_score > 0:
        return find_similar(query, top_k=top_k, exclude=exclude, min_score=0.0)

    return results


def generate_rationale(server: dict, project_description: str) -> str:
    """
    Template-based rationale: why this server fits this project.
    No LLM call — grounded, fast, offline.
    """
    name = server["name"]
    desc = server["description"]
    category = server["category"]
    score = server.get("score", 0)

    # Extract key nouns from project description (simple word overlap)
    proj_words = set(re.findall(r"[a-z0-9]+", project_description.lower()))
    desc_words = set(re.findall(r"[a-z0-9]+", desc.lower()))
    overlap = (proj_words & desc_words) - STOPWORDS

    if overlap:
        match_hint = f"It shares focus on: {', '.join(sorted(overlap)[:5])}."
    else:
        match_hint = f"It falls under the '{category}' category, which aligns with your project's needs."

    confidence = "strong" if score > 0.55 else "moderate" if score > 0.40 else "potential"

    return (
        f"{name} — {desc} "
        f"[{confidence} match | category: {category}] "
        f"{match_hint}"
    )


def lookup_by_name(server_name: str) -> dict | None:
    """Find a server by name, ranked: exact → short-name exact → prefix → substring."""
    if not server_name or not server_name.strip():
        return None
    q = server_name.strip().lower()
    con = get_connection()
    rows = con.execute(
        "SELECT name, description, url, category FROM servers WHERE lower(name) LIKE ?",
        [f"%{q}%"],
    ).fetchall()
    con.close()
    if not rows:
        return None

    def _rank(row):
        n = row[0].lower()
        parts = n.split("/", 1)
        owner = parts[0] if len(parts) == 2 else ""
        short = parts[-1]
        if n == q or short == q:
            return (0, 0)
        # owner exact match is a stronger prefix signal
        if owner == q:
            return (1, 0)
        if n.startswith(q) or short.startswith(q):
            return (1, 1)
        return (2, 0)

    name, desc, url, cat = min(rows, key=_rank)
    return {"name": name, "description": desc, "url": url, "category": cat}
