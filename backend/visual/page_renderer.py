"""PDF page rendering helpers for visual retrieval."""

from PIL import Image
import fitz

from backend.pdf_utils import quiet_mupdf

quiet_mupdf()


def render_page_as_image(pdf_path, page_number, dpi=150) -> Image.Image:
    """Render a 1-based PDF page number as an RGB PIL image."""

    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(f"Page {page_number} is outside the PDF page range.")

        page = document.load_page(page_number - 1)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
