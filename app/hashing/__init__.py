"""
app/hashing - Canonical JSON serialization and cryptographic hashing module.
"""

from app.hashing.canonical import (
    CanonicalBuilder,
    DETERMINISTIC_SCHEMA_KEYS,
    normalize_iso8601_utc,
    normalize_text_nfc,
    serialize_canonical_json,
    strip_volatile_fields,
)
from app.hashing.hasher import (
    CryptographicHasher,
    compute_image_sha256,
    compute_keccak256_digest,
    compute_sha256_digest,
)

__all__ = [
    "CanonicalBuilder",
    "CryptographicHasher",
    "DETERMINISTIC_SCHEMA_KEYS",
    "compute_image_sha256",
    "compute_keccak256_digest",
    "compute_sha256_digest",
    "normalize_iso8601_utc",
    "normalize_text_nfc",
    "serialize_canonical_json",
    "strip_volatile_fields",
]
