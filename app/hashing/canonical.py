"""
app/hashing/canonical.py - Deterministic Canonical Metadata Normalization & Serialization.

Implements Requirement R4:
- Unicode UTF-8 NFC normalization.
- ISO-8601 UTC timestamp formatting (YYYY-MM-DDTHH:MM:SSZ).
- Alphabetical key sorting at all nesting levels.
- Whitespace trimming on all strings.
- Volatile / non-deterministic field stripping.
- Deterministic JSON serialization: json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=False).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import unicodedata

from app.config import config
from app.models import CanonicalMetadata, CanonicalMetadataResult, SearchCandidate


DETERMINISTIC_SCHEMA_KEYS: Set[str] = {
    "author",
    "caption",
    "media_sha256",
    "post_id",
    "post_timestamp",
    "schema_version",
    "search_provider",
    "similarity_score",
    "source_url",
    "validated_at",
}


def normalize_text_nfc(text: Any) -> Any:
    """
    Normalizes string to Unicode NFC format and trims leading/trailing whitespace.
    If input is not a string, returns as is.
    """
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFC", text).strip()


def normalize_iso8601_utc(ts_input: Union[str, datetime, int, float]) -> str:
    """
    Parses timestamp in any standard format / timezone and normalizes to ISO-8601 UTC:
    YYYY-MM-DDTHH:MM:SSZ
    """
    if isinstance(ts_input, (int, float)):
        # Treat as unix timestamp in seconds
        dt = datetime.fromtimestamp(float(ts_input), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if isinstance(ts_input, datetime):
        if ts_input.tzinfo is None:
            dt_utc = ts_input.replace(tzinfo=timezone.utc)
        else:
            dt_utc = ts_input.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    if not isinstance(ts_input, str):
        raise ValueError(f"Invalid timestamp type: {type(ts_input)}")

    ts_str = ts_input.strip()
    if not ts_str:
        raise ValueError("Timestamp string cannot be empty")

    # Handle Z suffix
    if ts_str.endswith("Z"):
        clean_str = ts_str[:-1] + "+00:00"
    elif "z" in ts_str:
        clean_str = ts_str.replace("z", "+00:00")
    else:
        clean_str = ts_str

    # Attempt parsing with datetime.fromisoformat
    try:
        dt = datetime.fromisoformat(clean_str)
    except ValueError:
        # Fallback to standard formats
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %Z",
        ]
        parsed_dt = None
        for fmt in date_formats:
            try:
                parsed_dt = datetime.strptime(ts_str, fmt)
                break
            except ValueError:
                continue
        if parsed_dt is None:
            raise ValueError(f"Unable to parse timestamp: {ts_input}")
        dt = parsed_dt

    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    # Format without microseconds for determinism: YYYY-MM-DDTHH:MM:SSZ
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_volatile_fields(data: Dict[str, Any], allowed_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Strips non-deterministic/volatile runtime fields (keys starting with '_' or ephemeral keys).
    If allowed_keys is provided, retains only matching keys.
    """
    clean_dict: Dict[str, Any] = {}
    for k, v in data.items():
        k_clean = str(k).strip()
        # Skip volatile fields starting with '_'
        if k_clean.startswith("_"):
            continue
        # Skip ephemeral runtime keys
        if k_clean in {"temp_path", "execution_pid", "runtime_ms", "ephemeral_id"}:
            continue
        if allowed_keys is not None and k_clean not in allowed_keys:
            continue
        clean_dict[k_clean] = v
    return clean_dict


def serialize_canonical_json(data: Dict[str, Any], float_precision: int = 6) -> Tuple[str, bytes]:
    """
    Serializes a dictionary into canonical deterministic JSON according to R4 rules:
    - Unicode NFC normalization
    - Trimmed whitespace on keys and values
    - Recursively sorted keys
    - Standard separators (',', ':')
    - Rounded floating point precision to avoid IEEE 754 platform variances
    - UTF-8 bytes output
    """
    def _normalize_obj(obj: Any) -> Any:
        if isinstance(obj, str):
            return normalize_text_nfc(obj)
        elif isinstance(obj, dict):
            return {normalize_text_nfc(k): _normalize_obj(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [_normalize_obj(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, float_precision)
        return obj

    normalized = _normalize_obj(data)
    json_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return json_str, json_str.encode("utf-8")


class CanonicalBuilder:
    """
    Assembles, normalizes, and serializes canonical metadata from pipeline execution data.
    """

    def __init__(self, output_file: Optional[Path] = None):
        self.output_file = output_file or config.paths.canonical_post_file

    def build(
        self,
        candidate_or_data: Union[SearchCandidate, Dict[str, Any]],
        search_provider: Optional[str] = None,
        similarity_score: Optional[Union[float, str]] = None,
        media_sha256: Optional[str] = None,
        validated_at: Optional[Union[str, datetime]] = None,
        output_file: Optional[Path] = None,
    ) -> CanonicalMetadataResult:
        """
        Constructs a CanonicalMetadataResult from candidate data and validation metrics.
        """
        target_path = output_file or self.output_file

        if isinstance(candidate_or_data, SearchCandidate):
            raw_author = candidate_or_data.author or "Unknown"
            raw_caption = candidate_or_data.snippet or candidate_or_data.title or ""
            raw_source_url = candidate_or_data.source_url
            raw_post_id = str(candidate_or_data.raw_payload.get("post_id", f"post_{candidate_or_data.rank}"))
            raw_timestamp = candidate_or_data.post_date or datetime.now(timezone.utc).isoformat()
            prov_used = search_provider or "unknown_provider"
        elif isinstance(candidate_or_data, dict):
            raw_author = candidate_or_data.get("author", "Unknown")
            raw_caption = candidate_or_data.get("caption") or candidate_or_data.get("snippet") or candidate_or_data.get("title") or ""
            raw_source_url = candidate_or_data.get("source_url", "")
            raw_post_id = str(candidate_or_data.get("post_id", "post_001"))
            raw_timestamp = candidate_or_data.get("post_timestamp") or candidate_or_data.get("post_date") or candidate_or_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
            prov_used = search_provider or candidate_or_data.get("search_provider", "unknown_provider")
        else:
            raise TypeError(f"Unsupported candidate type: {type(candidate_or_data)}")

        # Normalize fields
        norm_author = normalize_text_nfc(raw_author)
        norm_caption = normalize_text_nfc(raw_caption)
        norm_source_url = normalize_text_nfc(raw_source_url)
        norm_post_id = normalize_text_nfc(raw_post_id)
        norm_timestamp = normalize_iso8601_utc(raw_timestamp)
        norm_provider = normalize_text_nfc(prov_used)

        # Handle media_sha256
        if media_sha256:
            norm_media_sha256 = normalize_text_nfc(media_sha256).lower()
        elif isinstance(candidate_or_data, dict) and "media_sha256" in candidate_or_data:
            norm_media_sha256 = normalize_text_nfc(candidate_or_data["media_sha256"]).lower()
        else:
            # Default to empty SHA-256 digest if not supplied
            norm_media_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        # Handle similarity score
        if similarity_score is not None:
            if isinstance(similarity_score, (int, float)):
                final_sim: Union[str, float] = round(float(similarity_score), 4)
            else:
                final_sim = str(similarity_score).strip()
        elif isinstance(candidate_or_data, dict) and "similarity_score" in candidate_or_data:
            sim_val = candidate_or_data["similarity_score"]
            final_sim = round(float(sim_val), 4) if isinstance(sim_val, (int, float)) else str(sim_val).strip()
        else:
            final_sim = 1.0

        # Handle validated_at timestamp
        norm_validated_at: Optional[str] = None
        if validated_at is not None:
            norm_validated_at = normalize_iso8601_utc(validated_at)
        elif isinstance(candidate_or_data, dict) and candidate_or_data.get("validated_at"):
            norm_validated_at = normalize_iso8601_utc(candidate_or_data["validated_at"])

        canonical_dict: Dict[str, Any] = {
            "author": norm_author,
            "caption": norm_caption,
            "media_sha256": norm_media_sha256,
            "post_id": norm_post_id,
            "post_timestamp": norm_timestamp,
            "schema_version": "1.0.0",
            "search_provider": norm_provider,
            "similarity_score": final_sim,
            "source_url": norm_source_url,
        }
        if norm_validated_at is not None:
            canonical_dict["validated_at"] = norm_validated_at

        # Validate with Pydantic model
        canonical_obj = CanonicalMetadata(**canonical_dict)

        # Serialize canonically
        json_str, canonical_bytes = serialize_canonical_json(canonical_dict)

        # Save to disk
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json_str, encoding="utf-8")

        return CanonicalMetadataResult(
            canonical_obj=canonical_obj,
            canonical_dict=canonical_dict,
            canonical_json_bytes=canonical_bytes,
            canonical_json_str=json_str,
            canonical_file_path=target_path,
        )
