"""Query rewriting for weak retrieval results."""

from backend.generation.prompt import (
    INTERNAL_SYSTEM_PROMPT,
    build_query_rewrite_prompt,
)


def rewrite_query(llm, original_query) -> str:
    """Rewrite a query to improve retrieval specificity."""

    prompt = build_query_rewrite_prompt(original_query)
    rewritten = llm.generate(prompt, system_prompt=INTERNAL_SYSTEM_PROMPT).strip()
    return _clean_rewrite(rewritten) or original_query


def _clean_rewrite(text) -> str:
    """Clean common wrapper text from a rewritten query."""

    text = text.strip().strip('"').strip("'")
    for prefix in ("Rewritten query:", "Query:", "Search query:"):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip().strip('"').strip("'")
    return text
