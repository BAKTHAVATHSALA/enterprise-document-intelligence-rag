"""Docling PDF Layout Parser Utility.

Parses PDF documents using Docling (or fallback PDF layout engine),
extracting structured layout elements: pages, headings, paragraphs, and tables.
"""

import os
import re
import tempfile
from typing import NamedTuple
from utils.logger import logger

# Try importing Docling DocumentConverter for static analysis & execution
try:
    from docling.document_converter import DocumentConverter  # type: ignore # noqa: F401
    DOCLING_AVAILABLE: bool = True
except Exception:
    DocumentConverter = None  # type: ignore
    DOCLING_AVAILABLE = False

# Try importing pypdf for layout fallback
try:
    from pypdf import PdfReader  # type: ignore # noqa: F401
    PYPDF_AVAILABLE: bool = True
except Exception:
    PdfReader = None  # type: ignore
    PYPDF_AVAILABLE = False

FUNC_PARSE_PDF: str = "parse_pdf_document"


class ParsedPageBlock(NamedTuple):
    """Structured block element extracted from PDF page layout."""
    page: int
    section: str
    text: str
    is_heading: bool
    is_table: bool


class ParsedDocumentResult(NamedTuple):
    """Container output from PDF layout extraction."""
    filename: str
    total_pages: int
    blocks: list[ParsedPageBlock]
    full_text: str


def parse_pdf_document(pdf_bytes: bytes, filename: str) -> ParsedDocumentResult:
    """Parse PDF file bytes into structured pages, headings, paragraphs, and tables.

    @param pdf_bytes: Raw binary content of uploaded PDF file.
    @param filename: Name of source PDF file.
    @returns: ParsedDocumentResult containing structured layout blocks and full text.
    @raises ValueError: If PDF content is empty or unparseable.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Cannot parse empty PDF file (0 bytes).")

    # Validate PDF Magic Header (%PDF-)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid file content: File does not have a valid PDF header (%PDF-).")

    logger.info(FUNC_PARSE_PDF, f"Parsing PDF document '{filename}' ({len(pdf_bytes)} bytes)")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        temp_file.write(pdf_bytes)
        temp_file.flush()
        temp_path: str = temp_file.name
        temp_file.close()

        blocks: list[ParsedPageBlock] = []
        full_text_parts: list[str] = []
        page_count: int = 1

        # 1. Primary Attempt: Docling DocumentConverter Parser
        docling_success: bool = False
        if DOCLING_AVAILABLE and DocumentConverter is not None:
            try:
                converter = DocumentConverter()
                conversion_result = converter.convert(temp_path)
                docling_doc = conversion_result.document

                if hasattr(docling_doc, "pages") and docling_doc.pages:
                    page_count = len(docling_doc.pages)

                current_section: str = "General"
                for item, level in docling_doc.iterate_items():
                    item_text: str = str(getattr(item, "text", "")).strip()
                    if not item_text:
                        continue

                    label: str = str(getattr(item, "label", "")).lower()
                    is_heading: bool = "heading" in label or "title" in label or "header" in label
                    is_table: bool = "table" in label

                    if is_heading:
                        current_section = item_text

                    page_no: int = 1
                    prov = getattr(item, "provenance", None)
                    if prov and isinstance(prov, list) and len(prov) > 0:
                        page_no = getattr(prov[0], "page_no", 1)
                    elif hasattr(item, "page_no"):
                        page_no = getattr(item, "page_no", 1)

                    page_count = max(page_count, page_no)

                    blocks.append(
                        ParsedPageBlock(
                            page=page_no,
                            section=current_section,
                            text=item_text,
                            is_heading=is_heading,
                            is_table=is_table,
                        )
                    )
                    full_text_parts.append(item_text)

                docling_success = len(blocks) > 0
                if docling_success:
                    logger.info(FUNC_PARSE_PDF, f"Docling successfully parsed '{filename}' with {len(blocks)} elements across {page_count} pages.")
            except Exception as docling_err:
                logger.warning(FUNC_PARSE_PDF, f"Docling parser execution note/fallback: {docling_err}")

        # 2. Fallback Attempt: pypdf layout parser
        if not docling_success:
            if not PYPDF_AVAILABLE or PdfReader is None:
                raise ValueError(f"Neither Docling nor pypdf parser modules are available to process '{filename}'.")

            try:
                reader = PdfReader(temp_path)
                page_count = len(reader.pages)

                for page_idx, page in enumerate(reader.pages):
                    page_num: int = page_idx + 1
                    page_text: str = page.extract_text() or ""
                    if not page_text or not page_text.strip():
                        continue

                    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                    if not paragraphs:
                        paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]

                    current_sec: str = f"Page {page_num}"
                    for para in paragraphs:
                        lines = [l.strip() for l in para.split("\n") if l.strip()]
                        if not lines:
                            continue

                        first_line: str = lines[0]
                        is_h: bool = len(first_line) < 60 and (
                            re.match(r"^\d+[\.\s]+", first_line) is not None
                            or first_line.isupper()
                            or "clause" in first_line.lower()
                            or "section" in first_line.lower()
                            or "parties" in first_line.lower()
                            or "termination" in first_line.lower()
                            or "payment" in first_line.lower()
                            or "confidentiality" in first_line.lower()
                            or "overview" in first_line.lower()
                            or "retrieval" in first_line.lower()
                            or "security" in first_line.lower()
                            or "evaluation" in first_line.lower()
                        )

                        if is_h:
                            current_sec = first_line

                        blocks.append(
                            ParsedPageBlock(
                                page=page_num,
                                section=current_sec,
                                text=para,
                                is_heading=is_h,
                                is_table=False,
                            )
                        )
                        full_text_parts.append(para)

                logger.info(FUNC_PARSE_PDF, f"Layout parser extracted {len(blocks)} blocks across {page_count} pages.")
            except Exception as pdf_err:
                raise ValueError(f"Failed to parse PDF document '{filename}': {pdf_err}") from pdf_err

        if not blocks:
            raise ValueError(f"PDF document '{filename}' contains no readable text or structural elements.")

        full_text: str = "\n\n".join(full_text_parts)
        return ParsedDocumentResult(
            filename=filename,
            total_pages=page_count,
            blocks=blocks,
            full_text=full_text,
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
