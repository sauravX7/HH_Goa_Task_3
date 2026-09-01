"""
app/verification/comparator.py - Compares local canonical post data against on-chain smart contract records.

Implements Requirement R6:
- Compares local canonical data vs on-chain records.
- Verifies Keccak-256 and SHA-256 hash match.
- Identifies any mutated fields between local canonical representation and on-chain records.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from app.hashing.canonical import normalize_iso8601_utc, serialize_canonical_json
from app.hashing.hasher import compute_keccak256_digest, compute_sha256_digest

logger = logging.getLogger(__name__)


def _parse_iso_to_unix_timestamp(iso_str: str) -> Optional[int]:
    """Converts ISO-8601 UTC timestamp string to integer Unix epoch seconds."""
    try:
        norm = normalize_iso8601_utc(iso_str)
        # e.g., '2026-09-01T12:00:00Z'
        clean = norm[:-1] + "+00:00" if norm.endswith("Z") else norm
        dt = datetime.fromisoformat(clean)
        return int(dt.timestamp())
    except Exception as e:
        logger.debug(f"Could not convert ISO timestamp '{iso_str}' to unix: {e}")
        return None


def compare_canonical_vs_onchain(
    canonical_data: Dict[str, Any],
    on_chain_record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compares local canonical post dictionary against on-chain record returned by getPost().
    Computes Keccak-256 and SHA-256 digests and checks field-level consistency.
    """
    _, canonical_bytes = serialize_canonical_json(canonical_data)
    computed_keccak = compute_keccak256_digest(canonical_bytes).lower()
    computed_sha256 = compute_sha256_digest(canonical_bytes).lower()

    if not on_chain_record:
        return {
            "is_match": False,
            "hashes_match": False,
            "fields_match": False,
            "computed_keccak": computed_keccak,
            "computed_sha256": computed_sha256,
            "on_chain_content_hash": None,
            "mismatched_fields": [],
            "on_chain_record": None,
            "rationale": "No on-chain record found for comparison.",
        }

    raw_onchain_hash = str(on_chain_record.get("contentHash") or on_chain_record.get("content_hash") or "").lower()
    if raw_onchain_hash and not raw_onchain_hash.startswith("0x"):
        raw_onchain_hash = "0x" + raw_onchain_hash

    hashes_match = (computed_keccak == raw_onchain_hash)
    mismatches: List[Dict[str, Any]] = []

    # Compare key fields stored on-chain
    # 1. source_url
    local_source_url = canonical_data.get("source_url", "")
    onchain_source_url = on_chain_record.get("sourceUrl", on_chain_record.get("source_url", ""))
    if local_source_url != onchain_source_url:
        mismatches.append({
            "field": "source_url",
            "canonical_value": local_source_url,
            "on_chain_value": onchain_source_url,
            "reason": f"Source URL divergence: local '{local_source_url}' != on-chain '{onchain_source_url}'",
        })

    # 2. search_provider
    local_provider = canonical_data.get("search_provider", "")
    onchain_provider = on_chain_record.get("provider", on_chain_record.get("searchProvider", on_chain_record.get("search_provider", "")))
    if local_provider != onchain_provider:
        mismatches.append({
            "field": "search_provider",
            "canonical_value": local_provider,
            "on_chain_value": onchain_provider,
            "reason": f"Provider divergence: local '{local_provider}' != on-chain '{onchain_provider}'",
        })

    # 3. author
    local_author = canonical_data.get("author", "")
    onchain_author = on_chain_record.get("author", "")
    if local_author != onchain_author:
        mismatches.append({
            "field": "author",
            "canonical_value": local_author,
            "on_chain_value": onchain_author,
            "reason": f"Author divergence: local '{local_author}' != on-chain '{onchain_author}'",
        })

    # 4. post_id
    local_post_id = str(canonical_data.get("post_id", ""))
    onchain_post_id = str(on_chain_record.get("postId", on_chain_record.get("post_id", "")))
    if local_post_id != onchain_post_id:
        mismatches.append({
            "field": "post_id",
            "canonical_value": local_post_id,
            "on_chain_value": onchain_post_id,
            "reason": f"Post ID divergence: local '{local_post_id}' != on-chain '{onchain_post_id}'",
        })

    # 5. post_timestamp
    local_ts_str = canonical_data.get("post_timestamp", "")
    local_unix = _parse_iso_to_unix_timestamp(local_ts_str) if isinstance(local_ts_str, str) else local_ts_str
    onchain_ts = on_chain_record.get("postTimestamp", on_chain_record.get("post_timestamp", 0))
    if local_unix is not None and onchain_ts != 0 and local_unix != onchain_ts:
        mismatches.append({
            "field": "post_timestamp",
            "canonical_value": local_ts_str,
            "canonical_unix": local_unix,
            "on_chain_value": onchain_ts,
            "reason": f"Timestamp divergence: local epoch {local_unix} != on-chain timestamp {onchain_ts}",
        })

    fields_match = (len(mismatches) == 0)
    is_match = (hashes_match and fields_match)

    rationale = "Verification passed: All cryptographic digests and on-chain metadata match exactly." if is_match else (
        f"Verification failed: {len(mismatches)} field mismatch(es) detected." if hashes_match else
        f"Cryptographic hash mismatch: computed '{computed_keccak}' != on-chain '{raw_onchain_hash}'."
    )

    return {
        "is_match": is_match,
        "hashes_match": hashes_match,
        "fields_match": fields_match,
        "computed_keccak": computed_keccak,
        "computed_sha256": computed_sha256,
        "on_chain_content_hash": raw_onchain_hash,
        "mismatched_fields": mismatches,
        "on_chain_record": on_chain_record,
        "rationale": rationale,
    }


class VerificationComparator:
    """
    Comparator engine for validating local canonical metadata against on-chain PostRecord structures.
    """

    @staticmethod
    def compare(
        canonical_data: Dict[str, Any],
        on_chain_record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Runs complete comparison between canonical dictionary and on-chain record."""
        return compare_canonical_vs_onchain(canonical_data, on_chain_record)

    @staticmethod
    def verify_hash_integrity(
        canonical_data: Dict[str, Any],
        expected_keccak: str,
        expected_sha256: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Verifies whether canonical_data produces the expected Keccak-256 and SHA-256 hashes.
        Returns (is_valid, computed_keccak, computed_sha256).
        """
        _, canonical_bytes = serialize_canonical_json(canonical_data)
        computed_keccak = compute_keccak256_digest(canonical_bytes).lower()
        computed_sha256 = compute_sha256_digest(canonical_bytes).lower()

        exp_keccak = expected_keccak.lower()
        if not exp_keccak.startswith("0x"):
            exp_keccak = "0x" + exp_keccak

        keccak_valid = (computed_keccak == exp_keccak)
        sha_valid = True
        if expected_sha256:
            exp_sha = expected_sha256.lower().strip()
            sha_valid = (computed_sha256 == exp_sha)

        return (keccak_valid and sha_valid, computed_keccak, computed_sha256)
