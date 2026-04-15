"""
Anonymizer
Consistent name hashing, PII stripping, and data record anonymization.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def anonymize_name(name: str, salt: str = "cogniteam") -> str:
    """
    Deterministically map a real name to an anonymous ID.
    Same name + salt always produces the same output.
    The mapping is not reversible without the salt.
    """
    digest = hashlib.sha256(f"{salt}:{name.strip().lower()}".encode()).hexdigest()[:10]
    return f"Emp_{digest}"


def strip_email_addresses(text: str) -> str:
    return _EMAIL_RE.sub("[EMAIL REDACTED]", text)


def strip_phone_numbers(text: str) -> str:
    return _PHONE_RE.sub("[PHONE REDACTED]", text)


def strip_ssn(text: str) -> str:
    return _SSN_RE.sub("[SSN REDACTED]", text)


def strip_urls(text: str) -> str:
    return _URL_RE.sub("[URL REDACTED]", text)


def strip_all_pii(text: str) -> str:
    """Apply all PII stripping operations to a text string."""
    text = strip_email_addresses(text)
    text = strip_phone_numbers(text)
    text = strip_ssn(text)
    return text.strip()


def anonymize_record(record: dict[str, Any], name_fields: list[str] | None = None) -> dict[str, Any]:
    """
    Anonymize a data record dict:
    - Hash values in name_fields
    - Strip PII from all string values

    Args:
        record:      Input data dict.
        name_fields: List of field names to anonymize (hash).

    Returns:
        New dict with anonymized/stripped values.
    """
    name_fields = name_fields or ["name", "full_name", "employee_name", "sender", "receiver"]
    result = {}
    for key, value in record.items():
        if key in name_fields and isinstance(value, str):
            result[key] = anonymize_name(value)
        elif isinstance(value, str):
            result[key] = strip_all_pii(value)
        elif isinstance(value, dict):
            result[key] = anonymize_record(value, name_fields)
        elif isinstance(value, list):
            result[key] = [
                anonymize_record(item, name_fields) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result
