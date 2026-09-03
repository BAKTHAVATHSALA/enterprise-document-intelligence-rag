"""Document Helper Orchestrator.

Orchestrates PDF layout parsing via Docling, PII redaction, entity extraction,
and metadata-enriched layout-aware semantic chunking.
"""

import uuid
from datetime import datetime, timezone
from typing import NamedTuple
from interfaces.document_interface import (
    ChunkInterface,
    DocumentMetadataInterface,
    PIIMatchInterface,
    EntityInterface,
)
from utils.pdf_parser import parse_pdf_document, ParsedDocumentResult, ParsedPageBlock
from utils.pii_redactor import redact_pii, RedactionResult
from utils.entity_extractor import extract_entities
from utils.chunker import create_semantic_chunks, create_structured_chunks
from utils.logger import logger

FUNC_PROCESS_DOCUMENT: str = "process_document_pipeline"
FUNC_PROCESS_PDF: str = "process_pdf_pipeline"


class ProcessedDocumentResult(NamedTuple):
    """Container output from document ingestion pipeline."""
    metadata: DocumentMetadataInterface
    chunks: list[ChunkInterface]
    pii_matches: list[PIIMatchInterface]
    entities: list[EntityInterface]


def process_pdf_pipeline(
    pdf_bytes: bytes,
    filename: str,
    title: str | None = None,
    doc_id: str | None = None,
) -> ProcessedDocumentResult:
    """Run real PDF ingestion pipeline using Docling, PII redaction, entities, and chunking.

    @param pdf_bytes: Raw binary content of uploaded PDF file.
    @param filename: Name of original PDF file.
    @param title: Document title string.
    @param doc_id: Optional explicit document identifier.
    @returns: ProcessedDocumentResult containing document metadata, chunks, PII matches, and entities.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Cannot process empty PDF content (0 bytes).")

    document_id: str = doc_id if doc_id else f"doc_{uuid.uuid4().hex[:12]}"
    doc_title: str = title if title else filename
    logger.info(FUNC_PROCESS_PDF, f"Starting PDF ingestion pipeline for doc_id: {document_id} ('{filename}')")

    # 1. Real PDF Layout Parsing via Docling
    parsed_pdf: ParsedDocumentResult = parse_pdf_document(pdf_bytes=pdf_bytes, filename=filename)

    # 2. PII Redaction Step
    redaction_res: RedactionResult = redact_pii(parsed_pdf.full_text)
    
    # Apply PII redaction to individual page layout blocks
    redacted_blocks: list[ParsedPageBlock] = []
    for block in parsed_pdf.blocks:
        block_redaction = redact_pii(block.text)
        redacted_blocks.append(
            ParsedPageBlock(
                page=block.page,
                section=block.section,
                text=block_redaction.cleaned_text,
                is_heading=block.is_heading,
                is_table=block.is_table,
            )
        )

    # 3. Entity Extraction Step (from PII-redacted text)
    entities: list[EntityInterface] = extract_entities(redaction_res.cleaned_text)
    entity_names: list[str] = [e.text for e in entities]

    # 4. Structured Layout-Aware Chunking Step
    all_chunks: list[ChunkInterface] = create_structured_chunks(
        blocks=redacted_blocks,
        document_id=document_id,
        source_filename=filename,
        entities=entity_names,
    )

    # 5. Document Metadata Construction
    doc_metadata: DocumentMetadataInterface = DocumentMetadataInterface(
        document_id=document_id,
        title=doc_title,
        source_filename=filename,
        total_pages=parsed_pdf.total_pages,
        total_chunks=len(all_chunks),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        FUNC_PROCESS_PDF,
        f"Completed PDF pipeline for {document_id}: {parsed_pdf.total_pages} pages, {len(all_chunks)} chunks, {len(redaction_res.matches)} PII matches, {len(entities)} entities."
    )

    return ProcessedDocumentResult(
        metadata=doc_metadata,
        chunks=all_chunks,
        pii_matches=redaction_res.matches,
        entities=entities,
    )


def process_document_pipeline(
    raw_text: str,
    title: str,
    source_filename: str,
    doc_id: str | None = None,
) -> ProcessedDocumentResult:
    """Run full document understanding pipeline on raw text input.

    @param raw_text: Raw unparsed document string content.
    @param title: Title of document.
    @param source_filename: Name of original document file.
    @param doc_id: Optional explicit document identifier.
    @returns: ProcessedDocumentResult containing document metadata, chunks, PII matches, and entities.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot process empty document text")

    document_id: str = doc_id if doc_id else f"doc_{uuid.uuid4().hex[:12]}"
    logger.info(FUNC_PROCESS_DOCUMENT, f"Starting ingestion pipeline for doc_id: {document_id}")

    # 1. PII Redaction Step
    redaction_res: RedactionResult = redact_pii(raw_text)
    cleaned_text: str = redaction_res.cleaned_text

    # 2. Entity Extraction Step
    entities: list[EntityInterface] = extract_entities(cleaned_text)
    entity_names: list[str] = [e.text for e in entities]

    # 3. Layout-aware Semantic Chunking Step
    pages_text: list[str] = [p for p in cleaned_text.split("--- PAGE BREAK ---") if p.strip()]
    if not pages_text:
        pages_text = [cleaned_text]

    all_chunks: list[ChunkInterface] = []
    for page_num, page_content in enumerate(pages_text, start=1):
        page_chunks: list[ChunkInterface] = create_semantic_chunks(
            text=page_content,
            document_id=document_id,
            source_filename=source_filename,
            page=page_num,
            section="Main Body",
            entities=entity_names,
        )
        all_chunks.extend(page_chunks)

    # 4. Construct Document Metadata
    doc_metadata: DocumentMetadataInterface = DocumentMetadataInterface(
        document_id=document_id,
        title=title,
        source_filename=source_filename,
        total_pages=len(pages_text),
        total_chunks=len(all_chunks),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        FUNC_PROCESS_DOCUMENT,
        f"Completed pipeline for {document_id}: {len(all_chunks)} chunks, {len(redaction_res.matches)} PII matches, {len(entities)} entities."
    )

    return ProcessedDocumentResult(
        metadata=doc_metadata,
        chunks=all_chunks,
        pii_matches=redaction_res.matches,
        entities=entities,
    )
