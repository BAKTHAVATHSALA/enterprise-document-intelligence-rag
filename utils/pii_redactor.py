"""PII Detection and Redaction Utility.

Detects and redacts sensitive PII information (SSN, Email, Phone, Credit Cards, API Keys)
to ensure sensitive enterprise data is protected prior to downstream AI processing.
"""

import re
from typing import NamedTuple
from interfaces.document_interface import PIIMatchInterface
from utils.logger import logger

# Function name constant for logging
FUNC_REDACT_PII: str = "redact_pii"

# Regex Patterns for Sensitive Data
SSN_PATTERN: str = r"\b\d{3}-\d{2}-\d{4}\b"
EMAIL_PATTERN: str = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE_PATTERN: str = r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
API_KEY_PATTERN: str = r"\b(?:sk|api|key|secret|token)_[a-zA-Z0-9]{16,}\b"
CREDIT_CARD_PATTERN: str = r"\b(?:\d{4}[- ]?){3}\d{4}\b"

PII_RULES: list[tuple[str, str, str]] = [
    ("SSN", SSN_PATTERN, "[REDACTED_SSN]"),
    ("EMAIL", EMAIL_PATTERN, "[REDACTED_EMAIL]"),
    ("PHONE", PHONE_PATTERN, "[REDACTED_PHONE]"),
    ("API_KEY", API_KEY_PATTERN, "[REDACTED_API_KEY]"),
    ("CREDIT_CARD", CREDIT_CARD_PATTERN, "[REDACTED_CREDIT_CARD]"),
]


class RedactionResult(NamedTuple):
    """Container for redacted text and detected PII records."""
    cleaned_text: str
    matches: list[PIIMatchInterface]


def redact_pii(text: str) -> RedactionResult:
    """Detect and redact sensitive PII tokens from input text.

    @param text: Raw input text string.
    @returns: RedactionResult containing cleaned_text and list of PIIMatchInterface matches.
    """
    if not text or not text.strip():
        return RedactionResult(cleaned_text=text, matches=[])

    matches: list[PIIMatchInterface] = []
    current_text: str = text

    for pii_type, pattern, replacement in PII_RULES:
        for match in re.finditer(pattern, current_text):
            original_val: str = match.group(0)
            matches.append(
                PIIMatchInterface(
                    pii_type=pii_type,
                    original_text=original_val,
                    redacted_text=replacement,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

    # Perform replacement
    for pii_type, pattern, replacement in PII_RULES:
        current_text = re.sub(pattern, replacement, current_text)

    if matches:
        logger.info(FUNC_REDACT_PII, f"Redacted {len(matches)} PII matches from document text.")
    
    return RedactionResult(cleaned_text=current_text, matches=matches)
