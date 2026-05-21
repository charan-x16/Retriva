"""Query rewriting for weak retrieval results."""


def rewrite_query(llm, original_query) -> str:
    """Rewrite a query to improve retrieval specificity."""

    prompt = (
        "Internal retrieval query rewrite. This is not a final user answer.\n"
        "Rewrite this search query to be more specific and retrieve better results.\n"
        "Keep the user's intent unchanged.\n"
        "Return ONLY the rewritten query. Nothing else.\n\n"
        f"Original query: {original_query}"
    )

    rewritten = llm.generate(prompt).strip()
    return _clean_rewrite(rewritten) or original_query


def _clean_rewrite(text) -> str:
    """Clean common wrapper text from a rewritten query."""

    text = text.strip().strip('"').strip("'")
    for prefix in ("Rewritten query:", "Query:", "Search query:"):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip().strip('"').strip("'")
    return text

