"""Central prompt builders for Retriva generation, correction, and evaluation."""

SYSTEM_PROMPT = """
You are Retriva, a document-grounded QA assistant.

Answer only from the provided context. If evidence is missing or conflicting,
say so. Preserve exact names, values, dates, units, and labels.

Cite every factual sentence exactly as [Source: page X, filename]. Use only
sources shown in the context. Do not invent citations or add a sources section.

Write clean ChatGPT-style prose: direct first sentence, short paragraphs, and
bullets only for requested lists or multiple separate points.
""".strip()


INTERNAL_SYSTEM_PROMPT = """
You are Retriva's internal retrieval controller. Follow the requested output
format exactly. Return no explanations or extra text.
""".strip()


VISUAL_SYSTEM_PROMPT = """
You are Retriva's visual document QA assistant.

Answer only from the provided PDF page images and source map. Read visible
layout, headings, forms, tables, captions, stamps, signatures, and values as
evidence. If evidence is missing, say what is missing.

Cite every factual sentence exactly as [Source: page X, filename]. Cite only
pages in the source map. Do not mention retrieval scores or add a sources
section.

Write clean ChatGPT-style prose: direct first sentence, short paragraphs, and
bullets only for requested lists or multiple separate points.
""".strip()


RAGAS_JUDGE_SYSTEM_PROMPT = """
You are a strict RAG judge. Return only one decimal number from 0.0 to 1.0.
No words, markdown, JSON, or explanation.
""".strip()


def build_answer_prompt(query, chunks) -> str:
    """Build the task prompt for one grounded document question."""

    return (
        "Use the context to answer the question. Write only the final answer.\n"
        "If the context is insufficient, say the document context does not "
        "contain enough information.\n\n"
        "<retrieved_context>\n"
        f"{_format_context(chunks)}\n"
        "</retrieved_context>\n\n"
        "<user_question>\n"
        f"{query}\n"
        "</user_question>\n\n"
        "<final_answer>"
    )


def build_visual_user_content(query, image_items) -> list[dict]:
    """Build an OpenAI-compatible multimodal user prompt with page images."""

    page_map = "\n".join(
        (
            f"- Page {item['page']}, {item['source']} "
            f"(retrieval score: {_format_score(item.get('score'))})"
        )
        for item in image_items
    )
    content = [
        {
            "type": "text",
            "text": (
                "Answer using only the attached PDF page images.\n\n"
                "<question>\n"
                f"{query}\n"
                "</question>\n\n"
                "<source_page_map>\n"
                f"{page_map}\n"
                "</source_page_map>\n\n"
                "Use the actual filename in citations; never write the literal "
                "word 'filename'."
            ),
        }
    ]

    for item in image_items:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": item["data_url"]},
            }
        )
    return content


def build_context_grade_prompt(query, chunks) -> str:
    """Build a strict prompt for grading retrieved context quality."""

    return (
        "Score context usefulness for answering the question.\n"
        "Scale: 1.0 complete, 0.7 mostly enough, 0.4 partial/indirect, "
        "0.0 irrelevant. Return only the number.\n\n"
        "<question>\n"
        f"{query}\n"
        "</question>\n\n"
        "<context>\n"
        f"{_format_grade_context(chunks[:3])}\n"
        "</context>"
    )


def build_query_rewrite_prompt(original_query) -> str:
    """Build a prompt for rewriting weak retrieval queries."""

    return (
        "Rewrite this for document retrieval. Preserve intent and important "
        "entities, dates, numbers, and terms. Make vague wording searchable. "
        "Do not answer or add facts. Return only the rewritten query.\n\n"
        "<original_query>\n"
        f"{original_query}\n"
        "</original_query>"
    )


def build_ragas_metric_prompt(metric, question, answer, context, reference):
    """Build a focused fallback scoring prompt for one evaluation metric."""

    metric_instructions = {
        "faithfulness": (
            "Score whether every factual claim in the answer is supported by "
            "the retrieved context. Penalize unsupported claims, wrong values, "
            "or citations that do not support the sentence."
        ),
        "answer_relevancy": (
            "Score whether the answer directly addresses the user's question. "
            "Penalize generic, incomplete, off-topic, or evasive answers."
        ),
        "context_precision": (
            "Score whether the retrieved context is focused on information "
            "needed to answer the question. Penalize noisy or unrelated context."
        ),
        "context_recall": (
            "Score whether the retrieved context contains enough information "
            "to support the reference answer. Penalize missing critical facts."
        ),
    }
    instruction = metric_instructions.get(metric, "Score this RAG output.")
    return (
        f"Metric: {metric}\n"
        f"Judge rule: {instruction}\n"
        "Scale: 1.0 excellent, 0.7 usable, 0.4 weak, 0.0 failed.\n\n"
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>\n\n"
        "<reference>\n"
        f"{reference}\n"
        "</reference>\n\n"
        "<retrieved_context>\n"
        f"{context}\n"
        "</retrieved_context>\n\n"
        "Return only the score."
    )


def _format_context(chunks) -> str:
    """Format retrieved chunks for prompt context."""

    lines = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "unknown")
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        lines.append(
            f"[Chunk {index} | page {page} | source {source}]\n"
            f"{text}"
        )
    return "\n\n".join(lines)


def _format_grade_context(chunks) -> str:
    """Format top chunks for retrieval grading."""

    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "unknown")
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        parts.append(f"Chunk {index} | page {page} | source {source}\n{text}")
    return "\n\n".join(parts)


def _format_score(score) -> str:
    """Format an optional retrieval score for the visual prompt."""

    if score is None:
        return "unknown"
    return f"{float(score):.3f}"
