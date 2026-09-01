"""
app/hashing/hasher.py - Dual Cryptographic Digest Generator (SHA-256 and EVM Keccak-256).

Implements Requirement R4:
- SHA-256 cryptographic digest (64 lowercase hex characters).
- Keccak-256 digest matching EVM bytes32 format (66 lowercase hex characters with '0x' prefix).
- Persistence to artifacts/sha256.txt and artifacts/keccak256.txt.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.config import config
from app.hashing.canonical import serialize_canonical_json
from app.models import CryptographicDigestResult


def compute_sha256_digest(data: Union[bytes, str, Dict[str, Any]]) -> str:
    """
    Computes 64-character lowercase hex SHA-256 digest.
    If data is dict, it is first canonically serialized.
    If data is str, it is UTF-8 encoded.
    """
    if isinstance(data, dict):
        _, raw_bytes = serialize_canonical_json(data)
    elif isinstance(data, str):
        raw_bytes = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(data)
    else:
        raise TypeError(f"Cannot compute SHA-256 for data type: {type(data)}")

    return hashlib.sha256(raw_bytes).hexdigest().lower()


def compute_keccak256_digest(data: Union[bytes, str, Dict[str, Any]]) -> str:
    """
    Computes 66-character lowercase hex Keccak-256 digest with '0x' prefix matching EVM bytes32.
    If data is dict, it is first canonically serialized.
    If data is str, it is UTF-8 encoded.
    """
    if isinstance(data, dict):
        _, raw_bytes = serialize_canonical_json(data)
    elif isinstance(data, str):
        raw_bytes = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(data)
    else:
        raise TypeError(f"Cannot compute Keccak-256 for data type: {type(data)}")

    # Prefer web3.py implementation
    try:
        from web3 import Web3
        hex_result = Web3.keccak(raw_bytes).hex().lower()
        if not hex_result.startswith("0x"):
            hex_result = "0x" + hex_result
        return hex_result
    except Exception:
        pass

    # Fallback to PyCryptodome
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(raw_bytes)
        return "0x" + k.hexdigest().lower()
    except Exception:
        pass

    # Fallback to eth_utils
    try:
        from eth_utils import keccak as eth_keccak
        return "0x" + eth_keccak(raw_bytes).hex().lower()
    except Exception as exc:
        raise RuntimeError(f"No suitable Keccak-256 provider available: {exc}")


def compute_image_sha256(image_path: Union[str, Path]) -> str:
    """
    Computes SHA-256 checksum of an image or file on disk.
    """
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 calculation: {image_path}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class CryptographicHasher:
    """
    Dual Cryptographic Digest Engine producing SHA-256 and Keccak-256 artifacts.
    """

    def __init__(
        self,
        sha256_file: Optional[Path] = None,
        keccak256_file: Optional[Path] = None,
    ):
        self.sha256_file = sha256_file or config.paths.sha256_file
        self.keccak256_file = keccak256_file or config.paths.keccak256_file

    def generate_digests(
        self,
        canonical_data: Union[bytes, str, Dict[str, Any]],
        sha256_path: Optional[Path] = None,
        keccak256_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> CryptographicDigestResult:
        """
        Computes SHA-256 and Keccak-256 digests and persists them to disk.
        """
        target_sha256 = sha256_path or kwargs.get("sha256_output_file") or self.sha256_file
        target_keccak = keccak256_path or kwargs.get("keccak256_output_file") or self.keccak256_file

        sha256_hex = compute_sha256_digest(canonical_data)
        keccak256_hex = compute_keccak256_digest(canonical_data)

        # Write sha256.txt
        target_sha256.parent.mkdir(parents=True, exist_ok=True)
        target_sha256.write_text(sha256_hex + "\n", encoding="utf-8")

        # Write keccak256.txt
        target_keccak.parent.mkdir(parents=True, exist_ok=True)
        target_keccak.write_text(keccak256_hex + "\n", encoding="utf-8")

        return CryptographicDigestResult(
            sha256_hash=sha256_hex,
            keccak256_hash=keccak256_hex,
            sha256_path=target_sha256,
            keccak256_path=target_keccak,
        )

    # Method alias
    hash_canonical_data = generate_digests

    # Alias for compatibility
    hash_canonical_data = generate_digests
