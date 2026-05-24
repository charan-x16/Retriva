"""PDF inspection helpers for deciding when visual indexing is useful."""

import os
from pathlib import Path

import fitz


def analyze_pdf(file_path) -> dict:
    """Inspect a PDF and summarize text density and visual content."""

    source = Path(file_path).name
    min_text_chars = int(os.getenv("VISUAL_MIN_TEXT_CHARS_PER_PAGE", "120"))
    min_image_area_ratio = float(os.getenv("VISUAL_MIN_IMAGE_AREA_RATIO", "0.15"))

    page_summaries = []
    with fitz.open(file_path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            image_area_ratio, image_count = _image_area_ratio(page)
            drawing_count = _drawing_count(page)
            page_summaries.append(
                {
                    "page": index,
                    "text_chars": len(text),
                    "image_count": image_count,
                    "image_area_ratio": image_area_ratio,
                    "drawing_count": drawing_count,
                    "has_low_text": len(text) < min_text_chars,
                    "has_meaningful_image": image_area_ratio >= min_image_area_ratio,
                    "is_scanned_like": (
                        len(text) < min_text_chars and image_area_ratio >= 0.5
                    ),
                    "is_drawing_heavy": drawing_count >= 25 and len(text) < 500,
                }
            )

    page_count = len(page_summaries)
    low_text_pages = sum(1 for page in page_summaries if page["has_low_text"])
    meaningful_image_pages = sum(
        1 for page in page_summaries if page["has_meaningful_image"]
    )
    scanned_like_pages = sum(1 for page in page_summaries if page["is_scanned_like"])
    drawing_heavy_pages = sum(1 for page in page_summaries if page["is_drawing_heavy"])
    total_text_chars = sum(page["text_chars"] for page in page_summaries)

    return {
        "source": source,
        "pages": page_count,
        "total_text_chars": total_text_chars,
        "avg_text_chars_per_page": (
            round(total_text_chars / page_count, 1) if page_count else 0
        ),
        "low_text_pages": low_text_pages,
        "meaningful_image_pages": meaningful_image_pages,
        "scanned_like_pages": scanned_like_pages,
        "drawing_heavy_pages": drawing_heavy_pages,
        "page_summaries": page_summaries,
    }


def should_index_visual(analysis, text_result=None) -> tuple[bool, str]:
    """Decide whether ColPali indexing should run for a PDF."""

    mode = os.getenv("VISUAL_INDEX_MODE", "auto").lower()
    if mode == "always":
        return True, "VISUAL_INDEX_MODE=always"
    if mode == "never":
        return False, "VISUAL_INDEX_MODE=never"

    text_failed = bool(text_result and text_result.get("status") == "failed")
    if text_failed:
        return True, "text_index_failed"

    pages = max(1, int(analysis.get("pages", 0)))
    if analysis.get("scanned_like_pages", 0):
        return True, "scanned_or_image_pdf"
    if analysis.get("meaningful_image_pages", 0):
        return True, "meaningful_images_detected"
    if analysis.get("drawing_heavy_pages", 0):
        return True, "drawing_or_layout_heavy_pages"
    if analysis.get("low_text_pages", 0) >= max(1, pages // 2):
        return True, "low_text_density"

    return False, "text_pdf_without_meaningful_visual_content"


def _image_area_ratio(page) -> tuple[float, int]:
    """Estimate how much of a page is covered by embedded images."""

    page_area = max(float(page.rect.get_area()), 1.0)
    try:
        image_infos = page.get_image_info(xrefs=True)
    except Exception:
        image_infos = []
    image_area = 0.0
    for info in image_infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox)
        image_area += max(float(rect.get_area()), 0.0)
    return min(image_area / page_area, 1.0), len(image_infos)


def _drawing_count(page) -> int:
    """Return a best-effort count of vector drawings on a page."""

    try:
        return len(page.get_drawings())
    except Exception:
        return 0
