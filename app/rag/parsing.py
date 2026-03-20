"""
Document Parsing Utilities
Supports TXT, PDF, and DOCX extraction with basic corruption handling.
"""

from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _normalize_cell(cell: object) -> str:
    """Normalize a PDF table cell into a single-line string."""
    if cell is None:
        return ""
    text = str(cell).replace("\n", " ").strip()
    return " ".join(text.split())


def _format_table_for_text(table: list, page_number: int, table_index: int) -> str:
    """Render a detected table in a text format that preserves row/column boundaries."""
    rows = []
    for row in table or []:
        normalized_cells = [_normalize_cell(cell) for cell in (row or [])]
        if any(normalized_cells):
            rows.append(" | ".join(normalized_cells))

    if not rows:
        return ""

    header = f"[TABLE page={page_number} index={table_index}]"
    return "\n".join([header, *rows])


def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from a plain text file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1", errors="ignore")


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required to parse PDF files") from exc

    text_parts = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Keep normal page text for narrative content.
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)

            # Preserve table rows/columns for invoice/BOL-style documents.
            for table_index, table in enumerate(page.extract_tables() or [], start=1):
                table_block = _format_table_for_text(table, page_number, table_index)
                if table_block:
                    text_parts.append(table_block)
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required to parse DOCX files") from exc

    document = Document(str(file_path))
    paragraphs = [p.text for p in document.paragraphs if p.text]
    return "\n".join(paragraphs)


def parse_document(file_path: Path) -> Optional[str]:
    """
    Parse a supported document and return extracted text.

    Returns None if the file is unsupported, empty, or unreadable.
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    try:
        if suffix == ".txt":
            text = extract_text_from_txt(file_path)
        elif suffix == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif suffix == ".docx":
            text = extract_text_from_docx(file_path)
        else:
            return None
    except Exception:
        return None

    if not text or not text.strip():
        return None

    return text
