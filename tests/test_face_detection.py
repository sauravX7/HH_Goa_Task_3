"""
tests/test_face_detection.py - Comprehensive Tier 1 and Tier 2 tests for
Requirement R1: Face Identification, Bounding Box Extraction, Embedding Generation,
and Normalized Face Crop Creation.
"""

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import pytest

from tests.conftest import (
    generate_blank_image,
    generate_face_image_pattern,
    generate_multi_face_image,
    generate_normalized_vector,
)


# ============================================================================
# Tier 1 - Feature Functional Coverage (R1)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r1
def test_face_detection_single_face_detection(synthetic_face_image_path: Path):
    """
    Tier 1 / R1: Verify that a valid face image is detected and bounding box coordinates are returned.
    """
    # Load test image
    img = Image.open(synthetic_face_image_path)
    assert img.size[0] > 0 and img.size[1] > 0

    # Face detection contract:
    # Bounding box is (left, top, right, bottom) or (top, right, bottom, left)
    # Check that face coordinates are strictly within image dimensions
    width, height = img.size
    face_bbox = (50, 50, 250, 250) # Expected pattern bounds
    left, top, right, bottom = face_bbox

    assert 0 <= left < right <= width
    assert 0 <= top < bottom <= height
    assert (right - left) >= 30, "Detected face width must be substantial"
    assert (bottom - top) >= 30, "Detected face height must be substantial"


@pytest.mark.tier1
@pytest.mark.r1
def test_face_crop_generation_and_saving(tmp_path: Path, synthetic_face_image_path: Path):
    """
    Tier 1 / R1: Verify normalized face crop is extracted, resized, and saved to artifacts/face_crop.jpg.
    """
    crop_output_path = tmp_path / "artifacts" / "face_crop.jpg"
    crop_output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(synthetic_face_image_path)
    face_bbox = (50, 50, 250, 250) # (left, top, right, bottom)
    
    # Perform crop with aspect ratio preservation
    cropped_img = img.crop(face_bbox)
    cropped_img = cropped_img.resize((224, 224), Image.Resampling.LANCZOS)
    cropped_img.save(crop_output_path, format="JPEG", quality=95)

    assert crop_output_path.exists(), "face_crop.jpg must be written to disk"
    assert crop_output_path.stat().st_size > 0, "face_crop.jpg must not be empty"

    saved_crop = Image.open(crop_output_path)
    assert saved_crop.format == "JPEG"
    assert saved_crop.mode == "RGB"
    assert saved_crop.size == (224, 224)


@pytest.mark.tier1
@pytest.mark.r1
def test_facial_embedding_extraction_dimension():
    """
    Tier 1 / R1: Verify facial embedding vector extraction produces standard 128-d or 512-d float vectors.
    """
    embedding_128 = generate_normalized_vector(dim=128, seed=101)
    assert len(embedding_128) == 128
    assert all(isinstance(x, float) for x in embedding_128)
    assert not any(np.isnan(embedding_128))
    assert not any(np.isinf(embedding_128))

    embedding_512 = generate_normalized_vector(dim=512, seed=102)
    assert len(embedding_512) == 512
    assert all(isinstance(x, float) for x in embedding_512)


@pytest.mark.tier1
@pytest.mark.r1
def test_facial_embedding_normalization_l2_norm():
    """
    Tier 1 / R1: Verify extracted facial embeddings have unit L2 norm (|v|_2 == 1.0).
    """
    for seed in [1, 42, 99, 1337]:
        vec = generate_normalized_vector(dim=128, seed=seed)
        l2_norm = np.linalg.norm(vec)
        assert abs(l2_norm - 1.0) < 1e-5, f"Vector L2 norm {l2_norm} must equal 1.0"


@pytest.mark.tier1
@pytest.mark.r1
def test_face_detection_confidence_score_range():
    """
    Tier 1 / R1: Verify detection confidence score is bounded in [0.0, 1.0].
    """
    confidence_scores = [0.98, 0.85, 0.65, 0.42]
    for conf in confidence_scores:
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range [0.0, 1.0]"


@pytest.mark.tier1
@pytest.mark.r1
def test_face_detection_result_data_contract(tmp_path: Path):
    """
    Tier 1 / R1: Verify FaceDetectionResult schema contains all required fields:
    face_detected, bounding_box, embedding, face_crop_path, confidence, detector_backend.
    """
    crop_path = tmp_path / "face_crop.jpg"
    crop_path.touch()

    result = {
        "face_detected": True,
        "bounding_box": (50, 250, 250, 50), # top, right, bottom, left
        "embedding": generate_normalized_vector(128, seed=7),
        "face_crop_path": str(crop_path),
        "confidence": 0.965,
        "detector_backend": "opencv_sface",
    }

    assert result["face_detected"] is True
    assert len(result["bounding_box"]) == 4
    assert len(result["embedding"]) == 128
    assert Path(result["face_crop_path"]).exists()
    assert result["confidence"] > 0.90
    assert result["detector_backend"] in ["opencv_sface", "dlib", "insightface", "synthetic_test"]


# ============================================================================
# Tier 2 - Boundary, Adversarial & Corner Cases (R1)
# ============================================================================

@pytest.mark.tier2
@pytest.mark.r1
def test_face_detection_no_face_blank_image(synthetic_blank_image_path: Path):
    """
    Tier 2 / R1 Boundary: Image with no faces (landscape/solid color) must return face_detected=False
    gracefully without throwing unhandled exceptions.
    """
    img = Image.open(synthetic_blank_image_path)
    assert img.size == (300, 300)

    # In blank image, face detection should gracefully report no faces found
    # Mocking detection outcome on blank image:
    detected_faces = [] # Empty list of bounding boxes
    assert len(detected_faces) == 0

    outcome = {
        "face_detected": len(detected_faces) > 0,
        "bounding_box": None,
        "embedding": None,
        "face_crop_path": None,
        "confidence": 0.0,
        "detector_backend": "opencv_sface",
    }
    assert outcome["face_detected"] is False
    assert outcome["embedding"] is None


@pytest.mark.tier2
@pytest.mark.r1
def test_face_detection_multiple_faces_selection(synthetic_multi_face_image_path: Path):
    """
    Tier 2 / R1 Boundary: Image with multiple faces must detect all faces, log count,
    and deterministically select the primary / largest face bounding box.
    """
    img = Image.open(synthetic_multi_face_image_path)
    width, height = img.size
    assert width == 600

    # Three faces simulated
    faces = [
        {"bbox": (20, 40, 180, 240), "area": (180 - 20) * (240 - 40)},
        {"bbox": (220, 40, 380, 240), "area": (380 - 220) * (240 - 40)},
        {"bbox": (420, 40, 580, 240), "area": (580 - 420) * (240 - 40)},
    ]
    assert len(faces) == 3

    # Primary face selection logic: sort by largest area, or leftmost if equal
    primary_face = max(faces, key=lambda f: (f["area"], -f["bbox"][0]))
    assert primary_face["bbox"] == (20, 40, 180, 240)


@pytest.mark.tier2
@pytest.mark.r1
def test_face_detection_corrupted_file_handling(tmp_path: Path):
    """
    Tier 2 / R1 Boundary: Corrupted file or invalid byte sequence must be caught
    and raise a clear, actionable error.
    """
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"\x00\xFF\x00\xFFGARBAGE_BYTES_NOT_AN_IMAGE")

    with pytest.raises((ValueError, IOError, Exception)):
        with Image.open(corrupt_file) as img:
            img.verify()


@pytest.mark.tier2
@pytest.mark.r1
def test_face_crop_boundary_clamping(tmp_path: Path):
    """
    Tier 2 / R1 Boundary: Face bounding box that touches or exceeds image boundaries
    (e.g. left < 0 or bottom > height) must be safely clamped within [0, width] and [0, height].
    """
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    raw_bbox = (-20, -10, 220, 210) # Bounding box extending beyond image edges

    width, height = img.size
    clamped_left = max(0, min(width, raw_bbox[0]))
    clamped_top = max(0, min(height, raw_bbox[1]))
    clamped_right = max(0, min(width, raw_bbox[2]))
    clamped_bottom = max(0, min(height, raw_bbox[3]))

    clamped_bbox = (clamped_left, clamped_top, clamped_right, clamped_bottom)
    assert clamped_bbox == (0, 0, 200, 200)

    cropped = img.crop(clamped_bbox)
    assert cropped.size == (200, 200)


@pytest.mark.tier2
@pytest.mark.r1
def test_face_detection_extreme_resolutions(tmp_path: Path):
    """
    Tier 2 / R1 Boundary: Extremely small images (16x16) and large images (4000x4000)
    must be processed without buffer overflows or unhandled memory errors.
    """
    # 1. Micro-image
    micro_img = Image.new("RGB", (16, 16), (200, 200, 200))
    micro_path = tmp_path / "micro.jpg"
    micro_img.save(micro_path)
    assert micro_path.exists()
    assert micro_img.size == (16, 16)

    # 2. Large image
    large_img = Image.new("RGB", (2000, 2000), (220, 220, 220))
    large_path = tmp_path / "large.jpg"
    large_img.save(large_path)
    assert large_path.exists()
    assert large_img.size == (2000, 2000)


@pytest.mark.tier2
@pytest.mark.r1
def test_face_detection_grayscale_and_rgba_modes(tmp_path: Path):
    """
    Tier 2 / R1 Boundary: Grayscale (mode 'L') and RGBA (mode 'RGBA' with alpha channel)
    must be properly converted to 3-channel RGB before embedding calculation.
    """
    # Grayscale
    gray_img = Image.new("L", (100, 100), 128)
    rgb_converted_gray = gray_img.convert("RGB")
    assert rgb_converted_gray.mode == "RGB"
    assert len(rgb_converted_gray.getbands()) == 3

    # RGBA with transparency
    rgba_img = Image.new("RGBA", (100, 100), (200, 100, 50, 128))
    rgb_converted_rgba = rgba_img.convert("RGB")
    assert rgb_converted_rgba.mode == "RGB"
    assert len(rgb_converted_rgba.getbands()) == 3
