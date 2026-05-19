"""Prompt construction and citation parsing for grounded answers."""

import re

SOURCE_RE = re.compile(r"\[Source:\s*page\s+(\d+),\s*([^\]]+)\]")


def generate_answer(llm, query, chunks) -> dict:
    """Generate an answer from chunks and extract inline source citations."""

    context = _format_context(chunks)
    prompt = (
        "Answer the question using only the provided context.\n"
        "For each fact, add [Source: page X, <source filename>] inline.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    answer = llm.generate(prompt)
    citations = _extract_citations(answer)
    return {"answer": answer, "citations": citations}


def _format_context(chunks) -> str:
    """Format retrieved chunks for the LLM prompt."""

    lines = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"{index}. Page {chunk.get('page')}, {chunk.get('source')}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(lines)


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

