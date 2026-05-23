"""Multimodal answer generation from retrieved PDF page images."""

import base64
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SOURCE_RE = re.compile(r"\[Source:\s*page\s+(\d+),\s*([^\]]+)\]")


def generate_visual_answer(query, visual_results) -> dict:
    """Generate a citation-grounded answer from retrieved visual page images."""

    load_dotenv()
    image_items = _build_image_items(visual_results)
    if not image_items:
        return {
            "answer": (
                "I found visual page matches, but this visual index does not "
                "include saved page images yet. Re-ingest the PDF in Visual "
                "(ColPali) mode, then ask again."
            ),
            "citations": [],
            "answer_mode": "visual_missing_images",
        }

    client = _openrouter_client()
    response = client.chat.completions.create(
        model=_visual_model_name(),
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_content(query, image_items)},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer,
        "citations": _extract_citations(answer),
        "answer_mode": "visual_multimodal",
    }


def _openrouter_client() -> OpenAI:
    """Create an OpenRouter client for multimodal chat completions."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for visual answers.")

    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not base_url or base_url.lower() == "none":
        base_url = "https://openrouter.ai/api/v1"

    return OpenAI(api_key=api_key, base_url=base_url)


def _visual_model_name() -> str:
    """Return the configured multimodal OpenRouter model."""

    return (
        os.getenv("VISUAL_OPENROUTER_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "google/gemma-4-31b-it:free"
    )


def _build_image_items(visual_results) -> list[dict]:
    """Prepare top retrieved page images for the multimodal request."""

    max_images = int(os.getenv("VISUAL_ANSWER_MAX_IMAGES", "3"))
    image_items = []
    for result in visual_results[:max_images]:
        image_path = result.get("image_path")
        if not image_path or not Path(image_path).exists():
            continue

        image_items.append(
            {
                "page": result.get("page"),
                "source": result.get("source", "unknown"),
                "score": result.get("score"),
                "data_url": _image_data_url(image_path),
            }
        )
    return image_items


def _image_data_url(image_path) -> str:
    """Read a local page image as an OpenAI-compatible data URL."""

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _system_prompt() -> str:
    """Return the visual document QA system prompt."""

    return (
        "You are Retriva's visual document QA assistant. Answer only from the "
        "provided PDF page images. Be clear, concise, and specific.\n\n"
        "Readability rules:\n"
        "- Write in a natural ChatGPT-like style.\n"
        "- Prefer short readable paragraphs.\n"
        "- For simple fact questions, write one direct sentence.\n"
        "- Never use a bullet list for a one-sentence answer.\n"
        "- Use bullets only when the user asks for a list/key points or when "
        "there are several separate items.\n"
        "- Do not compress many facts into one long sentence.\n"
        "- Avoid unnecessary personal details unless the user asks for them.\n\n"
        "Grounding rules:\n"
        "- If the images do not contain enough evidence, say that clearly.\n"
        "- Cite every factual bullet or sentence with "
        "[Source: page X, filename].\n"
        "- Do not cite pages that were not provided."
    )


def _user_content(query, image_items) -> list[dict]:
    """Build a multimodal user message with question, page map, and images."""

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
                "Answer the question using only the attached retrieved PDF "
                "page images.\n\n"
                f"Question: {query}\n\n"
                f"Available source pages:\n{page_map}\n\n"
                "Write a clean readable answer. Prefer natural paragraphs. "
                "Use bullets only when they genuinely improve readability. "
                "Required citation format: [Source: page X, actual filename]. "
                "Do not include the literal word 'filename'."
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


def _format_score(score) -> str:
    """Format an optional retrieval score for the prompt."""

    if score is None:
        return "unknown"
    return f"{float(score):.3f}"


def _extract_citations(answer) -> list[dict]:
    """Extract unique [Source: page X, filename] citations from an answer."""

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
