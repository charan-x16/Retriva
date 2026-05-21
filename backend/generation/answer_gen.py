"""Answer generation and citation parsing for grounded responses."""

import re

from backend.generation.prompt import build_answer_prompt

SOURCE_RE = re.compile(r"\[Source:\s*page\s+(\d+),\s*([^\]]+)\]")


def generate_answer(llm, query, chunks) -> dict:
    """Generate an answer from chunks and extract inline source citations."""

    prompt = build_answer_prompt(query, chunks)
    answer = llm.generate(prompt)
    citations = _extract_citations(answer)
    return {"answer": answer, "citations": citations}


def _extract_citations(answer) -> list[dict]:
    """Extract unique [Source: page X, filename] citation tags."""

    citations = []
    seen = set()
    for match in SOURCE_RE.finditer(answer):
        page = int(match.group(1))
        source = match.group(2).strip()
        key = (page, source)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"page": page, "source": source})
    return citations
