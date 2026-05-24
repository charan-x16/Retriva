"""LLM-based grading for retrieved context quality."""

import re

from backend.generation.prompt import (
    INTERNAL_SYSTEM_PROMPT,
    build_context_grade_prompt,
)


def grade_context(llm, query, chunks) -> float:
    """Return a 0.0 to 1.0 score for how well chunks answer the query."""

    prompt = build_context_grade_prompt(query, chunks)
    response = llm.generate(prompt, system_prompt=INTERNAL_SYSTEM_PROMPT)
    try:
        return _parse_score(response)
    except ValueError:
        return 0.5


def _parse_score(text) -> float:
    """Extract and clamp a float score from LLM output."""

    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text.strip())
    if not match:
        raise ValueError("No score found.")

    score = float(match.group(0))
    return max(0.0, min(1.0, score))
