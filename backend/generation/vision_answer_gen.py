"""Multimodal answer generation from retrieved PDF page images."""

import base64
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.generation.prompt import VISUAL_SYSTEM_PROMPT, build_visual_user_content

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
            {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_visual_user_content(query, image_items),
            },
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
        or "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
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
