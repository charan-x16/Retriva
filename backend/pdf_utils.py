"""Small PDF helpers for quiet parsing and repair of malformed uploads."""

from pathlib import Path

import fitz


def quiet_mupdf() -> None:
    """Disable MuPDF console noise for PDFs that can still be repaired."""

    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)


def repair_pdf(input_path, output_path) -> str:
    """Write a cleaned PDF copy and return its path."""

    quiet_mupdf()
    output_path = Path(output_path)
    with fitz.open(input_path) as document:
        try:
            pdf_bytes = document.tobytes(garbage=4, clean=True, deflate=True)
        except TypeError:
            pdf_bytes = document.tobytes(garbage=4, deflate=True)

    output_path.write_bytes(pdf_bytes)
    return str(output_path)
