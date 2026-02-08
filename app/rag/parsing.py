"""
Document Parsing Utilities
Supports TXT, PDF, and DOCX extraction with basic corruption handling.
"""

from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


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
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
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
