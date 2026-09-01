"""
tests/test_adversarial_stress.py - Comprehensive Adversarial Stress Testing & Edge Case Suite.

Covers:
1. Image edge cases: corrupted files, zero bytes, truncated images, no faces, multi-face canvases.
2. Unicode edge cases: combining accents (NFC vs NFD), emojis, non-BMP astral chars, whitespace trimming.
3. Timestamp variations: timezone offsets (+05:30, -08:00, Z), epoch timestamps, date-only formats.
4. Extreme similarity thresholds: 0.00, 1.00, negative thresholds (-1.00, -0.50), thresholds > 1.0, zero vectors.
5. Tamper detection sensitivity: 100% detection rate on subtle 1-bit/1-char mutations, micro-timestamp shifts, 1000-iteration mutation fuzzing.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unicodedata
import cv2
import numpy as np
from PIL import Image
import pytest

from app.face.cropper import FaceCropper
from app.face.detector import DetectedFace, FaceDetector
from app.face.encoder import FaceEncoder
from app.hashing.canonical import (
    CanonicalBuilder,
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
from app.models import (
    CanonicalMetadata,
    SearchCandidate,
    ValidationStatus,
)
from app.tamper.differ import TamperDiffEngine
from app.tamper.engine import TamperSuiteRunner
from app.tamper.scenarios import (
    get_all_tamper_scenarios,
    mutate_caption,
    mutate_media_hash,
    mutate_remove_field,
    mutate_source_url,
    mutate_timestamp,
)
from app.validation.engine import MatchValidationEngine
from app.validation.metrics import (
    compute_cosine_similarity,
    compute_euclidean_distance,
    normalize_vector,
    synthesize_candidate_vector_with_similarity,
)
from app.verification.comparator import VerificationComparator, compare_canonical_vs_onchain
from tests.conftest import MockBlockchainRegistry


# ============================================================================
# 1. Adversarial Image Edge Cases
# ============================================================================

class TestAdversarialImageInputs:
    """Stress tests face detection, cropping, and encoding against malformed and abnormal images."""

    @pytest.fixture
    def detector(self):
        return FaceDetector(backend="auto")

    @pytest.fixture
    def cropper(self):
        return FaceCropper()

    @pytest.fixture
    def encoder(self):
        return FaceEncoder()

    def test_corrupted_zero_byte_file(self, tmp_path, detector):
        """Zero-byte file must raise a clear ValueError / FileNotFoundError without unhandled crash."""
        zero_file = tmp_path / "zero.jpg"
        zero_file.write_bytes(b"")

        with pytest.raises((ValueError, FileNotFoundError)):
            detector.detect_faces(zero_file)

    def test_truncated_jpeg_header_only(self, tmp_path, detector):
        """Truncated JPEG file (only SOI marker) must raise ValueError."""
        trunc_file = tmp_path / "truncated.jpg"
        trunc_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")

        with pytest.raises(ValueError):
            detector.detect_faces(trunc_file)

    def test_random_binary_garbage(self, tmp_path, detector):
        """Random binary junk file must raise ValueError."""
        junk_file = tmp_path / "garbage.jpg"
        junk_file.write_bytes(os.urandom(4096))

        with pytest.raises(ValueError):
            detector.detect_faces(junk_file)

    def test_text_file_disguised_as_image(self, tmp_path, detector):
        """HTML or plain text file named .jpg must raise ValueError."""
        txt_file = tmp_path / "fake.jpg"
        txt_file.write_text("<!DOCTYPE html><html><body>Not an image</body></html>")

        with pytest.raises(ValueError):
            detector.detect_faces(txt_file)

    def test_solid_color_and_random_noise_no_faces(self, detector):
        """Solid black, solid white, and random noise images must return 0 faces without errors."""
        # Solid black
        black_img = np.zeros((300, 300, 3), dtype=np.uint8)
        assert len(detector.detect_faces(black_img)) == 0

        # Solid white
        white_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        assert len(detector.detect_faces(white_img)) == 0

        # Uniform random noise
        noise_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        assert len(detector.detect_faces(noise_img)) == 0

    def test_tiny_and_empty_images(self, detector):
        """Images below min_face_size or empty arrays must return 0 faces or raise ValueError."""
        tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert detector.detect_faces(tiny_img) == []

        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect_faces(empty_img)

    def test_multi_face_ranking_and_clamping(self, detector, cropper):
        """Test multi-face detection selection strategies: largest, center, highest_confidence."""
        # Create synthetic detected faces with known areas and positions
        f1 = DetectedFace(bounding_box=(20, 80, 80, 20), confidence=0.80)   # area = 60*60 = 3600, center = (50, 50)
        f2 = DetectedFace(bounding_box=(100, 250, 250, 100), confidence=0.95) # area = 150*150 = 22500, center = (175, 175)
        f3 = DetectedFace(bounding_box=(40, 200, 90, 150), confidence=0.99)  # area = 50*50 = 2500, center = (175, 65)

        faces = [f1, f2, f3]

        # Largest strategy
        largest = detector.get_primary_face(faces, selection_strategy="largest")
        assert largest == f2
        assert largest.area == 22500

        # Highest confidence strategy
        most_confident = detector.get_primary_face(faces, selection_strategy="highest_confidence")
        assert most_confident == f3
        assert most_confident.confidence == 0.99

        # Center strategy (canvas 300x300, center at 150, 150)
        center_face = detector.get_primary_face(faces, selection_strategy="center", image_shape=(300, 300))
        # Distance of f2 center (175, 175) to (150, 150) is (25^2 + 25^2) = 1250
        # Distance of f1 center (50, 50) to (150, 150) is (100^2 + 100^2) = 20000
        # Distance of f3 center (175, 65) to (150, 150) is (25^2 + 85^2) = 7850
        assert center_face == f2

    def test_cropper_boundary_out_of_bounds_padding(self, tmp_path, cropper):
        """Cropper handles face bounding boxes near or exceeding image borders with black border padding."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Face located right at corner (top-left)
        edge_face = DetectedFace(bounding_box=(0, 30, 30, 0), confidence=0.9)

        crop_out = tmp_path / "edge_crop.jpg"
        cropped = cropper.crop_face(img, edge_face, padding_ratio=0.5, target_size=(256, 256), save_path=crop_out)

        assert cropped.shape == (256, 256, 3)
        assert crop_out.exists()
        assert crop_out.stat().st_size > 0


# ============================================================================
# 2. Unicode & Text Normalization Edge Cases
# ============================================================================

class TestAdversarialUnicodeNormalization:
    """Stress tests Unicode NFC normalization, astral characters, whitespace, and key sorting."""

    def test_decomposed_vs_composed_accents_identical_digests(self):
        """Decomposed accents (NFD: 'e' + U+0301) and precomposed accents (NFC: U+00E9) must yield bitwise identical digests."""
        nfc_caption = "Café au lait à Paris crème brûlée"
        nfd_caption = unicodedata.normalize("NFD", nfc_caption)

        assert nfc_caption != nfd_caption  # Raw strings differ in code points

        data1 = {"caption": nfc_caption, "author": "René Descartes"}
        data2 = {"caption": nfd_caption, "author": unicodedata.normalize("NFD", "René Descartes")}

        json1, bytes1 = serialize_canonical_json(data1)
        json2, bytes2 = serialize_canonical_json(data2)

        assert json1 == json2
        assert bytes1 == bytes2
        assert compute_sha256_digest(bytes1) == compute_sha256_digest(bytes2)
        assert compute_keccak256_digest(bytes1) == compute_keccak256_digest(bytes2)

    def test_astral_plane_emojis_and_special_scripts(self):
        """Astral plane emojis (multi-byte, surrogate pairs), ZWJ sequences, and mathematical symbols."""
        complex_text = "Alice 👩‍👩‍👧‍👦 🚀 🔥 𝕬𝖑𝖎𝖈𝖊 測試 한국어 العربية עִבְרִית 𝄞"
        norm = normalize_text_nfc(complex_text)
        assert unicodedata.is_normalized("NFC", norm)

        payload = {"author": complex_text, "caption": "Astral emoji post 🌟"}
        json_str, bytes_out = serialize_canonical_json(payload)
        assert isinstance(json_str, str)
        assert len(bytes_out) > 0
        # Re-parse to ensure valid JSON
        parsed = json.loads(json_str)
        assert parsed["author"] == unicodedata.normalize("NFC", complex_text)

    def test_whitespace_trimming_variations(self):
        """Leading/trailing whitespace, tabs, newlines, carriage returns must be trimmed."""
        messy_str = "\t\r\n   Alice Doe in Wonderland   \n\t\r "
        clean = normalize_text_nfc(messy_str)
        assert clean == "Alice Doe in Wonderland"

        data = {
            "author": "\n\t  Alice   \t",
            "caption": "  Deep Search Caption\r\n  ",
            "post_id": "   post_12345 \t",
        }
        _, bytes_out = serialize_canonical_json(data)
        parsed = json.loads(bytes_out.decode("utf-8"))
        assert parsed["author"] == "Alice"
        assert parsed["caption"] == "Deep Search Caption"
        assert parsed["post_id"] == "post_12345"

    def test_nested_key_sorting_determinism(self):
        """Key sorting at all levels regardless of insertion order."""
        dict_a = {"z": 1, "a": {"d": 4, "b": 2, "c": 3}, "m": [1, 2, 3]}
        dict_b = {"a": {"c": 3, "d": 4, "b": 2}, "m": [1, 2, 3], "z": 1}

        json_a, bytes_a = serialize_canonical_json(dict_a)
        json_b, bytes_b = serialize_canonical_json(dict_b)

        assert json_a == json_b
        assert bytes_a == bytes_b
        assert compute_keccak256_digest(bytes_a) == compute_keccak256_digest(bytes_b)


# ============================================================================
# 3. Timestamp & Timezone Variations
# ============================================================================

class TestAdversarialTimestampNormalization:
    """Stress tests timestamp parsing across different timezone offsets, unix epoch seconds, and formats."""

    def test_equivalent_timezones_produce_identical_utc(self):
        """Timestamps in IST (+05:30), PST (-08:00), UTC (Z) representing the same instant must produce identical UTC strings."""
        ts_utc = "2026-09-01T10:00:00Z"
        ts_ist = "2026-09-01T15:30:00+05:30"
        ts_pst = "2026-09-01T02:00:00-08:00"
        ts_cet = "2026-09-01T12:00:00+02:00"

        norm_utc = normalize_iso8601_utc(ts_utc)
        norm_ist = normalize_iso8601_utc(ts_ist)
        norm_pst = normalize_iso8601_utc(ts_pst)
        norm_cet = normalize_iso8601_utc(ts_cet)

        assert norm_utc == "2026-09-01T10:00:00Z"
        assert norm_ist == norm_utc
        assert norm_pst == norm_utc
        assert norm_cet == norm_utc

    def test_unix_epoch_integers_and_floats(self):
        """Unix epoch integer seconds and floats normalize to UTC ISO-8601."""
        epoch = 1788256800  # 2026-09-01T10:00:00Z
        norm_int = normalize_iso8601_utc(epoch)
        norm_float = normalize_iso8601_utc(float(epoch))

        assert norm_int == "2026-09-01T10:00:00Z"
        assert norm_float == "2026-09-01T10:00:00Z"

    def test_date_only_format(self):
        """Date-only 'YYYY-MM-DD' normalizes to midnight UTC."""
        date_str = "2026-09-01"
        norm = normalize_iso8601_utc(date_str)
        assert norm == "2026-09-01T00:00:00Z"

    def test_invalid_and_empty_timestamps_raise_value_error(self):
        """Invalid timestamp strings and empty strings must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_iso8601_utc("")

        with pytest.raises(ValueError):
            normalize_iso8601_utc("not-a-timestamp")

        with pytest.raises(ValueError):
            normalize_iso8601_utc("2026-02-31T12:00:00Z")  # Non-existent date


# ============================================================================
# 4. Extreme & Boundary Similarity Thresholds
# ============================================================================

class TestAdversarialSimilarityThresholds:
    """Stress tests MatchValidationEngine and vector metrics under extreme thresholds and vectors."""

    @pytest.fixture
    def sample_query_embedding(self):
        rng = np.random.RandomState(42)
        vec = rng.randn(128)
        return (vec / np.linalg.norm(vec)).tolist()

    def test_similarity_threshold_zero(self, sample_query_embedding):
        """Threshold 0.00 accepts orthogonal and positively correlated candidate vectors."""
        engine = MatchValidationEngine(similarity_threshold=0.00)
        # Synthesize vector with similarity 0.00
        cand_emb_orth = synthesize_candidate_vector_with_similarity(sample_query_embedding, 0.00, seed=1)
        cands = [SearchCandidate(rank=1, title="Orthogonal", source_url="https://example.com/1")]

        res = engine.evaluate_candidates(sample_query_embedding, cands, {1: cand_emb_orth})
        assert res["validation_status"] == "MATCH_CONFIRMED"
        assert res["selected_rank"] == 1
        assert abs(res["similarity_score"]) < 1e-4

    def test_similarity_threshold_one_exact_match(self, sample_query_embedding):
        """Threshold 1.00 only accepts identical vectors and rejects even 0.9999."""
        engine = MatchValidationEngine(similarity_threshold=1.00)
        exact_emb = list(sample_query_embedding)
        near_emb = synthesize_candidate_vector_with_similarity(sample_query_embedding, 0.99, seed=2)

        cands = [
            SearchCandidate(rank=1, title="Exact", source_url="https://example.com/exact"),
            SearchCandidate(rank=2, title="Near", source_url="https://example.com/near"),
        ]

        # When only near candidate is provided
        res_near = engine.evaluate_candidates(sample_query_embedding, [cands[1]], {2: near_emb})
        assert res_near["validation_status"] == "SIMILARITY_BELOW_THRESHOLD"
        assert res_near["selected_candidate"] is None

        # When exact match candidate is provided
        res_exact = engine.evaluate_candidates(sample_query_embedding, cands, {1: exact_emb, 2: near_emb})
        assert res_exact["validation_status"] == "MATCH_CONFIRMED"
        assert res_exact["selected_rank"] == 1
        assert abs(res_exact["similarity_score"] - 1.0) < 1e-4

    def test_negative_similarity_thresholds(self, sample_query_embedding):
        """Negative thresholds (e.g. -0.50) handle negatively aligned embeddings."""
        engine = MatchValidationEngine(similarity_threshold=-0.50)
        neg_emb = synthesize_candidate_vector_with_similarity(sample_query_embedding, -0.30, seed=3)
        cands = [SearchCandidate(rank=1, title="Negative Match", source_url="https://example.com/neg")]

        res = engine.evaluate_candidates(sample_query_embedding, cands, {1: neg_emb})
        assert res["validation_status"] == "MATCH_CONFIRMED"
        assert res["similarity_score"] >= -0.50

    def test_impossible_threshold_above_one(self, sample_query_embedding):
        """Threshold > 1.0 (e.g. 1.05) rejects all candidates gracefully without exceptions."""
        engine = MatchValidationEngine(similarity_threshold=1.05)
        cands = [SearchCandidate(rank=1, title="Match", source_url="https://example.com/1")]
        res = engine.evaluate_candidates(sample_query_embedding, cands, {1: sample_query_embedding})

        assert res["validation_status"] == "SIMILARITY_BELOW_THRESHOLD"
        assert res["selected_candidate"] is None
        assert len(res["rejected_candidates"]) == 1

    def test_degenerate_zero_and_nan_vectors(self):
        """Zero vectors or NaN vectors return 0.0 cosine similarity without ZeroDivisionError."""
        zero_vec = [0.0] * 128
        normal_vec = [1.0] + [0.0] * 127

        sim_zero = compute_cosine_similarity(zero_vec, normal_vec)
        assert sim_zero == 0.0

        sim_both_zero = compute_cosine_similarity(zero_vec, zero_vec)
        assert sim_both_zero == 0.0

        nan_vec = [float("nan")] * 128
        sim_nan = compute_cosine_similarity(nan_vec, normal_vec)
        assert sim_nan == 0.0


# ============================================================================
# 5. Tamper Detection Sensitivity & 100% Detection Rate Under Boundary Mutations
# ============================================================================

class TestAdversarialTamperSensitivity:
    """Stress tests the 5-scenario tamper engine against subtle, boundary-case, and fuzzing mutations."""

    @pytest.fixture
    def baseline_canonical_dict(self):
        return {
            "author": "Alice Doe",
            "caption": "Alice presenting at Web3 Summit 2026 in Lisbon",
            "media_sha256": "4a7d1ed414474e4033ac29ccb8653d9b13994e6378e9b6a9c4fb21a4f0b2f518",
            "post_id": "post_web3_98765",
            "post_timestamp": "2026-09-01T12:00:00Z",
            "schema_version": "1.0.0",
            "search_provider": "google_lens",
            "similarity_score": 0.9452,
            "source_url": "https://social.example.org/alice/posts/98765",
        }

    @pytest.fixture
    def mock_blockchain(self, baseline_canonical_dict):
        bc = MockBlockchainRegistry()
        _, canonical_bytes = serialize_canonical_json(baseline_canonical_dict)
        h = compute_keccak256_digest(canonical_bytes).lower()
        bc.register_post(
            content_hash=h,
            source_url=baseline_canonical_dict["source_url"],
            provider=baseline_canonical_dict["search_provider"],
            author=baseline_canonical_dict["author"],
            post_id=baseline_canonical_dict["post_id"],
            post_timestamp=1788264000,
        )
        return bc

    def test_all_5_standard_tamper_scenarios_100_percent_detection(self, baseline_canonical_dict, mock_blockchain):
        """Verifies 100% tamper detection across all 5 standard scenarios."""
        runner = TamperSuiteRunner(blockchain=mock_blockchain)
        report = runner.run_5_tamper_scenarios(baseline_canonical_dict)

        assert report["all_tampered_detected"] is True
        assert report["total_scenarios"] == 5
        assert report["detected_tamper_count"] == 5

        for scenario in report["scenarios"]:
            assert scenario["status"] == "TAMPER_DETECTED"
            assert scenario["hashes_differ"] is True
            assert len(scenario["diffs"]) >= 1

    def test_single_character_caption_mutation(self, baseline_canonical_dict, mock_blockchain):
        """Even a 1-character edit (e.g. adding a trailing period '.') must trigger TAMPER_DETECTED."""
        mutated = copy.deepcopy(baseline_canonical_dict)
        mutated["caption"] = baseline_canonical_dict["caption"] + "."

        runner = TamperSuiteRunner(blockchain=mock_blockchain)
        _, baseline_bytes = serialize_canonical_json(baseline_canonical_dict)
        orig_h = compute_keccak256_digest(baseline_bytes)

        outcome = runner._evaluate_tamper("SUBTLE_CAPTION", "1-char caption edit", baseline_canonical_dict, mutated, orig_h)
        assert outcome["status"] == "TAMPER_DETECTED"
        assert outcome["hashes_differ"] is True
        assert outcome["diffs"][0]["field_name"] == "caption"

    def test_micro_timestamp_shift_one_second(self, baseline_canonical_dict, mock_blockchain):
        """A 1-second timestamp offset (12:00:00Z -> 12:00:01Z) must trigger TAMPER_DETECTED."""
        mutated = copy.deepcopy(baseline_canonical_dict)
        mutated["post_timestamp"] = "2026-09-01T12:00:01Z"

        runner = TamperSuiteRunner(blockchain=mock_blockchain)
        _, baseline_bytes = serialize_canonical_json(baseline_canonical_dict)
        orig_h = compute_keccak256_digest(baseline_bytes)

        outcome = runner._evaluate_tamper("MICRO_TIMESTAMP", "1-sec timestamp shift", baseline_canonical_dict, mutated, orig_h)
        assert outcome["status"] == "TAMPER_DETECTED"
        assert outcome["hashes_differ"] is True
        assert outcome["diffs"][0]["field_name"] == "post_timestamp"

    def test_single_hex_nibble_media_hash_mutation(self, baseline_canonical_dict, mock_blockchain):
        """Flipping a single hex character in media_sha256 must trigger TAMPER_DETECTED."""
        orig_media_hash = baseline_canonical_dict["media_sha256"]
        # Change last char from '8' to '9'
        mutated_media_hash = orig_media_hash[:-1] + ("9" if orig_media_hash[-1] != "9" else "0")

        mutated = copy.deepcopy(baseline_canonical_dict)
        mutated["media_sha256"] = mutated_media_hash

        runner = TamperSuiteRunner(blockchain=mock_blockchain)
        _, baseline_bytes = serialize_canonical_json(baseline_canonical_dict)
        orig_h = compute_keccak256_digest(baseline_bytes)

        outcome = runner._evaluate_tamper("SUBTLE_HASH", "Single hex flip in media hash", baseline_canonical_dict, mutated, orig_h)
        assert outcome["status"] == "TAMPER_DETECTED"
        assert outcome["hashes_differ"] is True
        assert outcome["diffs"][0]["field_name"] == "media_sha256"

    def test_individual_field_deletion_matrix(self, baseline_canonical_dict, mock_blockchain):
        """Deleting any single field from the canonical payload must result in TAMPER_DETECTED."""
        runner = TamperSuiteRunner(blockchain=mock_blockchain)
        _, baseline_bytes = serialize_canonical_json(baseline_canonical_dict)
        orig_h = compute_keccak256_digest(baseline_bytes)

        for field_name in baseline_canonical_dict.keys():
            mutated = copy.deepcopy(baseline_canonical_dict)
            del mutated[field_name]

            outcome = runner._evaluate_tamper(f"DELETE_{field_name.upper()}", f"Deleted {field_name}", baseline_canonical_dict, mutated, orig_h)
            assert outcome["status"] == "TAMPER_DETECTED", f"Failed to detect deletion of {field_name}"
            assert outcome["hashes_differ"] is True
            assert any(d["field_name"] == field_name and d["tampered_value"] == "<MISSING>" for d in outcome["diffs"])

    def test_tamper_fuzzing_1000_mutations_100_percent_detection(self, baseline_canonical_dict, mock_blockchain):
        """High-volume mutation fuzzing: 1,000 randomized boundary mutations must yield 100.00% detection rate."""
        runner = TamperSuiteRunner(blockchain=mock_blockchain)
        _, baseline_bytes = serialize_canonical_json(baseline_canonical_dict)
        orig_h = compute_keccak256_digest(baseline_bytes)

        rng = np.random.RandomState(1337)
        total_mutations = 1000
        detected_count = 0

        keys = list(baseline_canonical_dict.keys())

        for i in range(total_mutations):
            mutated = copy.deepcopy(baseline_canonical_dict)
            mutation_type = rng.choice(["char_flip", "truncate", "append", "num_jitter", "protocol_flip", "delete"])
            target_key = rng.choice(keys)

            if mutation_type == "delete":
                mutated.pop(target_key, None)
            elif mutation_type == "append":
                mutated[target_key] = str(mutated[target_key]) + f"_mut_{i}"
            elif mutation_type == "truncate":
                val_str = str(mutated[target_key])
                mutated[target_key] = val_str[:-1] if len(val_str) > 1 else val_str + "x"
            elif mutation_type == "num_jitter":
                if target_key == "similarity_score":
                    mutated[target_key] = round(float(mutated[target_key]) + 0.0001 * (i % 10 + 1), 4)
                else:
                    mutated[target_key] = str(mutated[target_key]) + str(i)
            elif mutation_type == "protocol_flip":
                if "http" in str(mutated.get("source_url", "")):
                    mutated["source_url"] = mutated["source_url"].replace("https://", "http://")
                else:
                    mutated[target_key] = str(mutated[target_key]) + "_p"
            else:  # char_flip
                val_str = list(str(mutated[target_key]))
                if val_str:
                    idx = rng.randint(0, len(val_str))
                    val_str[idx] = chr((ord(val_str[idx]) + 1) % 128)
                    mutated[target_key] = "".join(val_str)

            # Evaluate tamper
            outcome = runner._evaluate_tamper(f"FUZZ_{i}", f"Fuzz mutation {i}", baseline_canonical_dict, mutated, orig_h)
            if outcome["status"] == "TAMPER_DETECTED" and outcome["hashes_differ"]:
                detected_count += 1

        detection_rate = (detected_count / total_mutations) * 100.0
        assert detection_rate == 100.0, f"Detection rate {detection_rate}% was not 100% ({detected_count}/{total_mutations})"
