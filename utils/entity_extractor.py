"""Entity Extraction Utility.

Identifies domain entities (Organizations, Contracts, Dates, People, Locations)
from document text to enrich chunk metadata and build knowledge graph nodes.
"""

import re
from interfaces.document_interface import EntityInterface
from utils.logger import logger

FUNC_EXTRACT_ENTITIES: str = "extract_entities"

# Regex Patterns for Domain Entity Extraction
ORG_PATTERN: str = r"\b[A-Z][A-Za-z0-9&\.\s]+(?:Corp|Corporation|Inc|LLC|Ltd|Group|Holdings|Bank|Co|Company|Industries|Enterprises|Technologies|Systems|Labs)\b"
CONTRACT_PATTERN: str = r"\b(?:Contract|Agreement|Policy|Document|Section|Clause)\s+(?:#[A-Za-z0-9-]+|[0-9]{3,}|[A-Z0-9_-]{4,})\b"
DATE_PATTERN: str = r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d+\s+days?)\b"
PERSON_TITLE_PATTERN: str = r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Director|Officer|Manager|President)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"

ENTITY_PATTERNS: list[tuple[str, str]] = [
    ("ORGANIZATION", ORG_PATTERN),
    ("CONTRACT", CONTRACT_PATTERN),
    ("DATE", DATE_PATTERN),
    ("PERSON", PERSON_TITLE_PATTERN),
]


def extract_entities(text: str) -> list[EntityInterface]:
    """Extract named domain entities from document text.

    @param text: Input document text string.
    @returns: List of extracted EntityInterface items.
    """
    if not text or not text.strip():
        return []

    entities: list[EntityInterface] = []
    seen_texts: set[str] = set()

    for category, pattern in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            extracted_str: str = match.group(0).strip()
            if extracted_str.lower() in seen_texts:
                continue

            seen_texts.add(extracted_str.lower())
            entities.append(
                EntityInterface(
                    text=extracted_str,
                    label=category,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.95,
                )
            )

    if entities:
        logger.info(FUNC_EXTRACT_ENTITIES, f"Extracted {len(entities)} entities from text.")

    return entities
