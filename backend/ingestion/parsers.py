"""PDF parsers for embedded text, scanned text, and tabular content."""

from io import BytesIO
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


def parse_text_pdf(file_path) -> list[dict]:
    """Extract embedded text from a PDF page by page."""

    source = Path(file_path).name
    docs = []
    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                docs.append({"text": text, "page": page_number, "source": source})
    return docs


def parse_scanned_pdf(file_path) -> list[dict]:
    """Render PDF pages to images and OCR them with Tesseract."""

    source = Path(file_path).name
    docs = []
    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image).strip()
            if text:
                docs.append({"text": text, "page": page_number, "source": source})
    return docs


def parse_tables(file_path) -> list[dict]:
    """Extract PDF tables with Camelot and convert them to markdown text."""

    source = Path(file_path).name
    tables = _read_tables(file_path)
    docs = []

    for table in tables:
        markdown = _dataframe_to_markdown(table.df)
        if not markdown:
            continue
        try:
            page = int(table.page)
        except (TypeError, ValueError):
            page = 0
        docs.append(
            {
                "text": markdown,
                "page": page,
                "source": source,
                "type": "table",
            }
        )
    return docs


def detect_and_parse(file_path) -> list[dict]:
    """Choose embedded-text parsing for digital PDFs or OCR for scans."""

    has_embedded_text = False
    with fitz.open(file_path) as pdf:
        for page in pdf:
            if page.get_text("text").strip():
                has_embedded_text = True
                break

    if has_embedded_text:
        return parse_text_pdf(file_path) + parse_tables(file_path)
    return parse_scanned_pdf(file_path)


def _read_tables(file_path):
    """Read tables with a simple lattice-then-stream fallback."""

    try:
        import camelot
    except ImportError:
        return []

    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(file_path, pages="all", flavor=flavor)
            if len(tables) > 0:
                return tables
        except Exception:
            continue
    return []


def _dataframe_to_markdown(df) -> str:
    """Convert a Camelot dataframe to a compact markdown table."""

    rows = df.fillna("").astype(str).values.tolist()
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]

    lines = [_markdown_row(header), _markdown_row(["---"] * width)]
    lines.extend(_markdown_row(row) for row in body)
    return "\n".join(lines)


def _markdown_row(values) -> str:
    """Format one markdown table row."""

    escaped = [
        str(value).replace("\n", " ").replace("|", "\\|").strip()
        for value in values
    ]
    return "| " + " | ".join(escaped) + " |"
