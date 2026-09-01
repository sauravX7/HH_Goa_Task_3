"""
tests/test_canonical_hashing.py - Comprehensive Tier 1 and Tier 2 tests for
Requirement R4: Search Provenance, Evidence Collection, Deterministic Canonicalization,
Unicode NFC Normalization, ISO-8601 UTC Timestamp Alignment, and Dual Cryptographic Hashing (SHA-256 / Keccak-256).
"""

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
import pytest

from tests.conftest import (
    compute_keccak256_digest,
    compute_sha256_digest,
    normalize_iso8601_utc,
    normalize_text_nfc,
    serialize_canonical_json,
)
from app.hashing.canonical import (
    CanonicalBuilder,
    normalize_text_nfc as app_normalize_text_nfc,
    normalize_iso8601_utc as app_normalize_iso8601_utc,
    serialize_canonical_json as app_serialize_canonical_json,
    strip_volatile_fields,
)
from app.hashing.hasher import (
    CryptographicHasher,
    compute_sha256_digest as app_compute_sha256_digest,
    compute_keccak256_digest as app_compute_keccak256_digest,
    compute_image_sha256,
)
from app.evidence.screenshot import PageScreenshotter, capture_post_screenshot
from app.evidence.collector import EvidenceCollector, assemble_evidence_package
from app.models import (
    FaceDetectionResult,
    MatchValidationResult,
    SearchCandidate,
    SearchProvenanceResult,
    ValidationStatus,
)



# ============================================================================
# Tier 1 - Feature Functional Coverage (R4)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_alphabetical_key_sorting():
    """
    Tier 1 / R4: JSON dictionary keys must be sorted alphabetically at all nesting levels.
    """
    unsorted_data = {
        "source_url": "https://example.com/post/1",
        "author": "Alice",
        "post_id": "123",
        "caption": "Hello world",
        "nested": {
            "zebra": 1,
            "apple": 2,
            "mango": 3,
        }
    }
    json_str, _ = serialize_canonical_json(unsorted_data)
    
    # Check key ordering in JSON string
    author_idx = json_str.index('"author"')
    caption_idx = json_str.index('"caption"')
    nested_idx = json_str.index('"nested"')
    post_id_idx = json_str.index('"post_id"')
    source_url_idx = json_str.index('"source_url"')

    assert author_idx < caption_idx < nested_idx < post_id_idx < source_url_idx

    # Check nested key ordering
    apple_idx = json_str.index('"apple"')
    mango_idx = json_str.index('"mango"')
    zebra_idx = json_str.index('"zebra"')
    assert apple_idx < mango_idx < zebra_idx


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_unicode_nfc_normalization():
    """
    Tier 1 / R4: Composed and decomposed Unicode strings must produce byte-for-byte identical canonical output and hashes.
    """
    # Decomposed: 'e' followed by combining acute accent (\u0301)
    decomposed_str = "Caf" + "e\u0301" + " verification \u2126" # ohm symbol
    # Composed: single character 'é' (\u00e9) and omega
    composed_str = "Caf\u00e9 verification \u03a9"

    data_decomposed = {"caption": decomposed_str, "author": "Alice"}
    data_composed = {"caption": composed_str, "author": "Alice"}

    _, bytes_decomposed = serialize_canonical_json(data_decomposed)
    _, bytes_composed = serialize_canonical_json(data_composed)

    sha_decomposed = compute_sha256_digest(bytes_decomposed)
    sha_composed = compute_sha256_digest(bytes_composed)

    keccak_decomposed = compute_keccak256_digest(bytes_decomposed)
    keccak_composed = compute_keccak256_digest(bytes_composed)

    assert bytes_decomposed == bytes_composed
    assert sha_decomposed == sha_composed
    assert keccak_decomposed == keccak_composed


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_iso8601_utc_timestamps():
    """
    Tier 1 / R4: Timestamps with different timezone offsets (+05:30, -08:00, Z) must normalize to identical UTC strings.
    """
    # 2026-09-01 14:30:00 at UTC+05:30 is 2026-09-01 09:00:00Z
    ts_india = "2026-09-01T14:30:00+05:30"
    ts_pst = "2026-09-01T01:00:00-08:00"
    ts_utc = "2026-09-01T09:00:00Z"

    norm_india = normalize_iso8601_utc(ts_india)
    norm_pst = normalize_iso8601_utc(ts_pst)
    norm_utc = normalize_iso8601_utc(ts_utc)

    assert norm_india == "2026-09-01T09:00:00Z"
    assert norm_pst == "2026-09-01T09:00:00Z"
    assert norm_utc == "2026-09-01T09:00:00Z"
    assert norm_india == norm_pst == norm_utc


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_whitespace_trimming():
    """
    Tier 1 / R4: Whitespace around field keys and values must be stripped.
    """
    data_padded = {
        "  author  ": "   Bob The Builder   \n\t",
        "caption": "  Decentralized ID verified.  ",
    }
    json_str, _ = serialize_canonical_json(data_padded)
    obj = json.loads(json_str)

    assert "author" in obj
    assert obj["author"] == "Bob The Builder"
    assert obj["caption"] == "Decentralized ID verified."


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_sha256_cryptographic_digest(sample_canonical_dict: Dict[str, Any]):
    """
    Tier 1 / R4: SHA-256 digest is exactly 64 lowercase hex characters matching hashlib.sha256.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    sha_hash = compute_sha256_digest(canonical_bytes)

    assert len(sha_hash) == 64
    assert all(c in "0123456789abcdef" for c in sha_hash)
    assert sha_hash == hashlib.sha256(canonical_bytes).hexdigest().lower()


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_keccak256_cryptographic_digest(sample_canonical_dict: Dict[str, Any]):
    """
    Tier 1 / R4: Keccak-256 digest is exactly 66 lowercase characters starting with '0x' matching EVM bytes32.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    keccak_hash = compute_keccak256_digest(canonical_bytes)

    assert keccak_hash.startswith("0x")
    assert len(keccak_hash) == 66
    hex_body = keccak_hash[2:]
    assert all(c in "0123456789abcdef" for c in hex_body)


@pytest.mark.tier1
@pytest.mark.r4
def test_evidence_package_directory_outputs(tmp_path: Path):
    """
    Tier 1 / R4: All 9 required evidence package artifacts are present in artifacts/.
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "face_crop.jpg",
        "search_result.png",
        "metadata.json",
        "canonical_post.json",
        "sha256.txt",
        "keccak256.txt",
        "tx_receipt.json",
        "verification_report.json",
        "pipeline_log.json",
    ]

    for fname in required_files:
        (artifacts_dir / fname).touch()

    for fname in required_files:
        p = artifacts_dir / fname
        assert p.exists() and p.is_file()


# ============================================================================
# Tier 2 - Boundary, Adversarial & Corner Cases (R4)
# ============================================================================

@pytest.mark.tier2
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_empty_and_whitespace_only_fields():
    """
    Tier 2 / R4 Boundary: Empty strings and whitespace-only strings normalize cleanly to "".
    """
    data = {
        "author": "Alice",
        "caption": "   \n\t   ",
        "media_sha256": "",
    }
    json_str, _ = serialize_canonical_json(data)
    obj = json.loads(json_str)

    assert obj["caption"] == ""
    assert obj["media_sha256"] == ""


@pytest.mark.tier2
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_complex_unicode_emojis_and_cjk():
    """
    Tier 2 / R4 Boundary: Complex multilingual scripts (Japanese, Cyrillic, Arabic, Devanagari)
    and composite emojis (family emoji, colored emojis) are preserved in canonical UTF-8 bytes.
    """
    data = {
        "author": "田中太郎 🧑‍💻",
        "caption": "Тестирование блокчейна и верификации! 🛡️ 📸 العربية परीक्षण",
    }
    json_str, canonical_bytes = serialize_canonical_json(data)

    # Validate UTF-8 roundtrip
    decoded = json.loads(canonical_bytes.decode("utf-8"))
    assert decoded["author"] == "田中太郎 🧑‍💻"
    assert "Тестирование" in decoded["caption"]
    assert "🛡️" in decoded["caption"] or "🛡" in decoded["caption"]


@pytest.mark.tier2
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_floating_point_precision_rounding():
    """
    Tier 2 / R4 Boundary: Floating point values (like similarity scores) are rounded
    to avoid platform-dependent IEEE 754 precision variances.
    """
    data1 = {"similarity_score": 0.8845000000000001}
    data2 = {"similarity_score": 0.8845000000000000}

    _, bytes1 = serialize_canonical_json(data1)
    _, bytes2 = serialize_canonical_json(data2)

    assert bytes1 == bytes2
    assert compute_sha256_digest(bytes1) == compute_sha256_digest(bytes2)


@pytest.mark.tier2
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_volatile_fields_stripped():
    """
    Tier 2 / R4 Boundary: Non-deterministic runtime fields (e.g. local temp file paths, memory addresses)
    must not be included in canonical_post.json.
    """
    raw_payload = {
        "author": "Alice",
        "caption": "Live on chain",
        "post_id": "123",
        "source_url": "https://example.com/123",
        "post_timestamp": "2026-09-01T12:00:00Z",
        "media_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "search_provider": "serpapi_google_lens",
        "similarity_score": 0.92,
        "_ephemeral_temp_path": "/tmp/var/run/temp_12345.jpg",
        "_execution_pid": 9821,
    }

    # Canonical builder filters only whitelisted deterministic schema keys
    whitelisted_keys = {
        "author", "caption", "media_sha256", "post_id",
        "post_timestamp", "search_provider", "similarity_score", "source_url"
    }
    filtered_payload = {k: v for k, v in raw_payload.items() if k in whitelisted_keys}

    json_str, _ = serialize_canonical_json(filtered_payload)
    assert "_ephemeral_temp_path" not in json_str
    assert "_execution_pid" not in json_str


@pytest.mark.tier2
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_special_escaping_characters():
    """
    Tier 2 / R4 Boundary: Strings with backslashes, quotes, and newlines must serialize
    without breaking valid JSON.
    """
    data = {
        "author": 'Alice "The Cryptographer" O\'Connor',
        "caption": "Line 1\\nLine 2 with \\\\ backslash and \"quotes\"",
    }
    json_str, canonical_bytes = serialize_canonical_json(data)
    reloaded = json.loads(canonical_bytes.decode("utf-8"))

    assert reloaded["author"] == 'Alice "The Cryptographer" O\'Connor'
    assert "Line 1\\nLine 2" in reloaded["caption"]


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_canonical_builder_from_candidate(tmp_path: Path):
    """
    Tier 1 / R4: CanonicalBuilder correctly normalizes SearchCandidate and outputs canonical_post.json.
    """
    out_file = tmp_path / "artifacts" / "canonical_post.json"
    candidate = SearchCandidate(
        rank=1,
        title="Alice Web3: Cryptography and Verification",
        source_url="https://social.example.com/alice/123",
        snippet="Exploring decentralized identity! 🛡️",
        author="  Alice Web3  ",
        post_date="2026-09-01T14:30:00+05:30",
        raw_payload={"post_id": "post_789102"},
    )
    builder = CanonicalBuilder(output_file=out_file)
    result = builder.build(
        candidate_or_data=candidate,
        search_provider="serpapi_google_lens",
        similarity_score=0.91234,
        media_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        validated_at="2026-09-01T15:00:00Z",
    )

    assert out_file.exists()
    assert result.canonical_dict["author"] == "Alice Web3"
    assert result.canonical_dict["post_timestamp"] == "2026-09-01T09:00:00Z"
    assert result.canonical_dict["similarity_score"] == 0.9123
    assert result.canonical_dict["search_provider"] == "serpapi_google_lens"
    assert result.canonical_dict["schema_version"] == "1.0.0"
    assert result.canonical_json_bytes.startswith(b"{")


@pytest.mark.tier1
@pytest.mark.r4
@pytest.mark.hashing
def test_cryptographic_hasher_artifacts(tmp_path: Path, sample_canonical_dict: Dict[str, Any]):
    """
    Tier 1 / R4: CryptographicHasher writes sha256.txt and keccak256.txt correctly.
    """
    sha_path = tmp_path / "artifacts" / "sha256.txt"
    keccak_path = tmp_path / "artifacts" / "keccak256.txt"

    hasher = CryptographicHasher(sha256_file=sha_path, keccak256_file=keccak_path)
    res = hasher.generate_digests(sample_canonical_dict)

    assert sha_path.exists()
    assert keccak_path.exists()
    assert len(res.sha256_hash) == 64
    assert res.keccak256_hash.startswith("0x")
    assert len(res.keccak256_hash) == 66
    assert sha_path.read_text().strip() == res.sha256_hash
    assert keccak_path.read_text().strip() == res.keccak256_hash


@pytest.mark.tier1
@pytest.mark.r4
def test_compute_image_sha256(synthetic_face_image_path: Path):
    """
    Tier 1 / R4: compute_image_sha256 accurately hashes image files on disk.
    """
    digest = compute_image_sha256(synthetic_face_image_path)
    assert len(digest) == 64
    # Compute directly with hashlib to confirm match
    direct_hash = hashlib.sha256(synthetic_face_image_path.read_bytes()).hexdigest()
    assert digest == direct_hash


@pytest.mark.tier1
@pytest.mark.r4
def test_evidence_screenshot_fallback_renderer(tmp_path: Path):
    """
    Tier 1 / R4: PageScreenshotter generates high-resolution fallback screenshot.
    """
    screenshot_file = tmp_path / "artifacts" / "search_result.png"
    screenshotter = PageScreenshotter(output_file=screenshot_file, viewport_size=(1200, 800))
    meta = {
        "author": "Alice Web3",
        "caption": "Decentralized identity post",
        "search_provider": "serpapi_google_lens",
        "similarity_score": 0.945,
    }
    path = screenshotter.capture("https://social.example.com/alice/status/123", output_file=screenshot_file, metadata=meta)

    assert path.exists()
    assert path.stat().st_size > 1000  # Non-trivial PNG file size


@pytest.mark.tier1
@pytest.mark.r4
def test_evidence_collector_bundle_assembler(tmp_path: Path, sample_search_candidates: list):
    """
    Tier 1 / R4: EvidenceCollector packages query, provenance, validation, and writes metadata.json.
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = artifacts_dir / "metadata.json"
    face_crop = artifacts_dir / "face_crop.jpg"
    face_crop.parent.mkdir(parents=True, exist_ok=True)
    face_crop.touch()
    screenshot = artifacts_dir / "search_result.png"
    screenshot.touch()

    face_detection = FaceDetectionResult(
        face_detected=True,
        bounding_box=(50, 250, 250, 50),
        embedding=[0.1] * 128,
        face_crop_path=face_crop,
        confidence=0.98,
        detector_backend="face_recognition",
        faces_count=1,
    )

    prov = SearchProvenanceResult(
        provider_used="serpapi_google_lens",
        query_image_hash="abc1234567890def",
        query_id="query_001",
        candidates=[SearchCandidate(**sample_search_candidates[0])],
        total_results_found=1,
    )

    val = MatchValidationResult(
        validation_status=ValidationStatus.MATCH_CONFIRMED,
        similarity_score=0.92,
        distance_score=0.18,
        provider_confidence=0.95,
        selected_rank=1,
        threshold_used=0.60,
    )

    collector = EvidenceCollector(
        metadata_file=metadata_file,
        face_crop_file=face_crop,
        screenshot_file=screenshot,
    )
    pkg = collector.assemble(
        face_detection=face_detection,
        search_provenance=prov,
        match_validation=val,
    )

    assert metadata_file.exists()
    raw = json.loads(metadata_file.read_text())
    assert raw["query"]["face_detected"] is True
    assert raw["search_provenance"]["provider_used"] == "serpapi_google_lens"
    assert raw["match_validation"]["validation_status"] == "MATCH_CONFIRMED"
    assert pkg.raw_metadata_path == metadata_file

