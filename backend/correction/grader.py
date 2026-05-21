"""LLM-based grading for retrieved context quality."""

import re


def grade_context(llm, query, chunks) -> float:
    """Return a 0.0 to 1.0 score for how well chunks answer the query."""

    context = _format_top_chunks(chunks[:3])
    prompt = (
        "Internal retrieval quality check. This is not a final user answer.\n"
        "Rate how well the following context answers this question.\n"
        "Return ONLY a number between 0.0 and 1.0. Nothing else.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )

    response = llm.generate(prompt)
    try:
        return _parse_score(response)
    except ValueError:
        return 0.5


def _format_top_chunks(chunks) -> str:
    """Join the top chunks into compact grading context."""

    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "unknown")
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        parts.append(f"Chunk {index} | page {page} | source {source}\n{text}")
    return "\n\n".join(parts)


def _parse_score(text) -> float:
    """Extract and clamp a float score from LLM output."""

    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text.strip())
    if not match:
        raise ValueError("No score found.")

    score = float(match.group(0))
    return max(0.0, min(1.0, score))

