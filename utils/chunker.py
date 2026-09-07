"""Semantic and Layout-Aware Document Chunker Utility.

Splits structured PDF blocks (from Docling layout parser) into semantic chunks,
preserving page numbers, section headers, entity lists, and source lineage.
"""

import hashlib
from typing import Optional
from interfaces import ChunkInterface, ChunkMetadataInterface
from utils.pdf_parser import ParsedPageBlock
from utils.logger import logger

FUNC_CHUNKER: str = "create_structured_chunks"
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 150


def generate_chunk_id(document_id: str, page: int, chunk_index: int) -> str:
    """Generate deterministic 16-character SHA-256 chunk identifier.

    @param document_id: Unique document identifier string.
    @param page: 1-indexed page number.
    @param chunk_index: 0-indexed sequential chunk index within document.
    @returns: 16-character hexadecimal SHA-256 hash string.
    """
    payload: str = f"{document_id}:{page}:{chunk_index}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]



def create_structured_chunks(
    blocks: list[ParsedPageBlock],
    document_id: str,
    source_filename: str,
    entities: list[str],
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ChunkInterface]:
    """Create structured layout-aware document chunks from Docling page blocks.

    Enforces strict page and section boundaries so chunks never cross pages.

    @param blocks: List of ParsedPageBlock objects from Docling layout extraction.
    @param document_id: Unique identifier of target document.
    @param source_filename: Filename of source PDF document.
    @param entities: List of extracted entity strings for metadata lineage.
    @param max_chunk_size: Maximum character length per chunk.
    @returns: List of ChunkInterface objects.
    """
    if not blocks:
        logger.warning(FUNC_CHUNKER, f"No blocks provided for chunking document '{document_id}'")
        return []

    chunks: list[ChunkInterface] = []
    current_blocks: list[ParsedPageBlock] = []
    current_len: int = 0
    chunk_counter: int = 0

    for block in blocks:
        if not block.text or not block.text.strip():
            continue

        # Boundary check: Flush chunk if page changes, if a new section heading starts, or if max length exceeded
        page_changed: bool = bool(current_blocks and block.page != current_blocks[0].page)
        heading_changed: bool = bool(current_blocks and block.is_heading and current_len > 80)
        size_exceeded: bool = bool(current_len + len(block.text) > max_chunk_size)

        if current_blocks and (page_changed or heading_changed or size_exceeded):
            chunk_text = "\n\n".join([b.text for b in current_blocks])
            chunk_page = current_blocks[0].page
            chunk_section = current_blocks[0].section or f"Page {chunk_page}"

            chunk_entities = [e for e in entities if e.lower() in chunk_text.lower()]
            if not chunk_entities:
                chunk_entities = entities[:5]

            chunk_id = generate_chunk_id(
                document_id=document_id,
                page=chunk_page,
                chunk_index=chunk_counter,
            )
            chunks.append(
                ChunkInterface(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    metadata=ChunkMetadataInterface(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page=chunk_page,
                        section=chunk_section,
                        source=source_filename,
                        entities=chunk_entities,
                    ),
                )
            )
            chunk_counter += 1
            current_blocks = []
            current_len = 0

        current_blocks.append(block)
        current_len += len(block.text)

    # Flush final remaining block
    if current_blocks:
        chunk_text = "\n\n".join([b.text for b in current_blocks])
        chunk_page = current_blocks[0].page
        chunk_section = current_blocks[0].section or f"Page {chunk_page}"

        chunk_entities = [e for e in entities if e.lower() in chunk_text.lower()]
        if not chunk_entities:
            chunk_entities = entities[:5]

        chunk_id = generate_chunk_id(
            document_id=document_id,
            page=chunk_page,
            chunk_index=chunk_counter,
        )
        chunks.append(
            ChunkInterface(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                metadata=ChunkMetadataInterface(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    page=chunk_page,
                    section=chunk_section,
                    source=source_filename,
                    entities=chunk_entities,
                ),
            )
        )

    logger.info(FUNC_CHUNKER, f"Generated {len(chunks)} structured chunks for doc: {document_id}")
    return chunks


def create_semantic_chunks(
    text: str,
    document_id: str,
    source_filename: str,
    page: int = 1,
    section: str = "General",
    entities: list[str] = [],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkInterface]:
    """Fallback sliding window chunker for unstructured text strings.

    @param text: Input text content.
    @param document_id: Unique document identifier string.
    @param source_filename: Source filename.
    @param page: 1-indexed page number (default 1).
    @param section: Section title string (default 'General').
    @param entities: List of extracted entity names.
    @param chunk_size: Target maximum chunk character length.
    @param overlap: Sliding window character overlap.
    @returns: List of ChunkInterface items.
    """
    if not text or not text.strip():
        return []

    chunks: list[ChunkInterface] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current_para: list[str] = []
    current_len: int = 0
    chunk_idx: int = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current_para:
            chunk_text = "\n\n".join(current_para)
            chunk_id = generate_chunk_id(
                document_id=document_id,
                page=page,
                chunk_index=chunk_idx,
            )
            chunks.append(
                ChunkInterface(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    metadata=ChunkMetadataInterface(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page=page,
                        section=section,
                        source=source_filename,
                        entities=entities,
                    ),
                )
            )
            chunk_idx += 1
            current_para = []
            current_len = 0

        current_para.append(para)
        current_len += len(para)

    if current_para:
        chunk_text = "\n\n".join(current_para)
        chunk_id = generate_chunk_id(
            document_id=document_id,
            page=page,
            chunk_index=chunk_idx,
        )
        chunks.append(
            ChunkInterface(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                metadata=ChunkMetadataInterface(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    page=page,
                    section=section,
                    source=source_filename,
                    entities=entities,
                ),
            )
        )

    return chunks
