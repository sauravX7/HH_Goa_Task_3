"""Unit tests for Face Identification, Feature Processing (M1 - R1), Configuration & Schemas (M0)."""

import math
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import AppSettings, load_config
from app.models import (
    BoundingBox,
    CanonicalMetadata,
    CryptographicDigestResult,
    FaceDetectionResult,
    MatchValidationResult,
    SearchCandidate,
    SearchProvenanceResult,
    StageExecutionLog,
    TamperDetectionResult,
    ValidationStatus,
    VerificationStatus,
)
from app.face.cropper import FaceCropper
from app.face.detector import DetectedFace, FaceDetector
from app.face.encoder import FaceEncoder


def create_synthetic_face_image(width: int = 400, height: int = 400) -> np.ndarray:
    """Create a synthetic RGB image with basic face-like geometry for testing."""
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    # Head outline (oval)
    center = (width // 2, height // 2)
    axes = (width // 4, height // 3)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 130), -1)

    # Eyes
    eye_y = height // 2 - height // 10
    left_eye_x = width // 2 - width // 8
    right_eye_x = width // 2 + width // 8
    cv2.circle(img, (left_eye_x, eye_y), width // 25, (50, 50, 50), -1)
    cv2.circle(img, (right_eye_x, eye_y), width // 25, (50, 50, 50), -1)

    # Nose
    nose_pts = np.array([
        [width // 2, eye_y + 10],
        [width // 2 - 10, eye_y + 40],
        [width // 2 + 10, eye_y + 40],
    ])
    cv2.fillPoly(img, [nose_pts], (150, 120, 100))

    # Mouth
    mouth_center = (width // 2, height // 2 + height // 8)
    mouth_axes = (width // 10, height // 25)
    cv2.ellipse(img, mouth_center, mouth_axes, 0, 0, 180, (100, 50, 50), 3)

    return img


# -------------------------------------------------------------------------
# M0: Configuration & Models Tests
# -------------------------------------------------------------------------

def test_config_loading():
    """Verify configuration loads properly with default values and types."""
    cfg = load_config()
    assert isinstance(cfg, AppSettings)
    assert cfg.app_name == "FaceProvenancePipeline"
    assert cfg.matching.similarity_threshold == 0.60
    assert cfg.blockchain.network == "hardhat"
    assert cfg.effective_network == "hardhat"
    assert cfg.effective_rpc_url == "http://127.0.0.1:8545"
    assert cfg.effective_chain_id == 31337
    assert cfg.paths.face_crop_file.name == "face_crop.jpg"
    assert cfg.paths.sha256_file.name == "sha256.txt"
    assert cfg.paths.keccak256_file.name == "keccak256.txt"


def test_pydantic_schemas_all_stages():
    """Verify core Pydantic v2 schemas for all stages can be instantiated and validated."""
    # Stage 1
    face_res = FaceDetectionResult(
        face_detected=True,
        bounding_box=(50, 200, 250, 50),
        embedding=[0.1] * 128,
        confidence=0.98,
        detector_backend="face_recognition",
        faces_count=1,
    )
    assert face_res.face_detected is True
    assert len(face_res.embedding) == 128

    # Stage 2
    cand = SearchCandidate(
        rank=1,
        title="Matching Social Post",
        source_url="https://example.com/post/123",
        provider_confidence=0.95,
    )
    prov = SearchProvenanceResult(
        provider_used="serpapi_lens",
        query_image_hash="abc12345",
        query_id="query_001",
        candidates=[cand],
    )
    assert prov.provider_used == "serpapi_lens"
    assert len(prov.candidates) == 1

    # Stage 3
    val = MatchValidationResult(
        validation_status=ValidationStatus.MATCH_CONFIRMED,
        similarity_score=0.885,
        distance_score=0.25,
        threshold_used=0.60,
    )
    assert val.validation_status == ValidationStatus.MATCH_CONFIRMED

    # Stage 5
    canon = CanonicalMetadata(
        author="Alice",
        caption="Verified selfie at summit",
        media_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        post_id="post_999",
        post_timestamp="2026-09-01T12:00:00Z",
        search_provider="serpapi_lens",
        similarity_score=0.92,
        source_url="https://example.com/post_999",
    )
    assert canon.schema_version == "1.0.0"

    # Stage 6
    digest = CryptographicDigestResult(
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        keccak256_hash="0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
        sha256_path=Path("artifacts/sha256.txt"),
        keccak256_path=Path("artifacts/keccak256.txt"),
    )
    assert digest.keccak256_hash.startswith("0x")


# -------------------------------------------------------------------------
# M1: Face Detector Tests
# -------------------------------------------------------------------------

def test_face_detector_no_face_blank_image():
    """Verify detector handles images with no faces gracefully without raising unhandled exceptions."""
    detector = FaceDetector(backend="face_recognition")
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detector.detect_faces(blank_img)
    assert isinstance(faces, list)
    assert len(faces) == 0

    primary = detector.get_primary_face(faces)
    assert primary is None


def test_face_detector_corrupt_input_raises_value_error():
    """Verify detector raises appropriate error on invalid image input."""
    detector = FaceDetector()
    with pytest.raises(ValueError):
        detector.detect_faces(np.array([]))

    with pytest.raises(ValueError):
        detector.detect_faces(b"not a valid image byte buffer")

    with pytest.raises(ValueError):
        detector.detect_faces("/non/existent/path/image.jpg")


def test_face_detector_geometry_properties():
    """Verify DetectedFace bounding box and geometric calculations."""
    face = DetectedFace(bounding_box=(50, 200, 250, 100), confidence=0.95)
    assert face.top == 50
    assert face.right == 200
    assert face.bottom == 250
    assert face.left == 100
    assert face.width == 100
    assert face.height == 200
    assert face.area == 20000
    assert face.center == (150.0, 150.0)

    bbox_model = face.to_bbox_model()
    assert isinstance(bbox_model, BoundingBox)
    assert bbox_model.top == 50
    assert bbox_model.width == 100


def test_multi_face_ranking_and_primary_selection():
    """Verify primary face selection strategies (largest, highest confidence, center)."""
    detector = FaceDetector()
    face_small = DetectedFace(bounding_box=(10, 50, 60, 10), confidence=0.99)    # area = 40 * 50 = 2000
    face_large = DetectedFace(bounding_box=(20, 280, 220, 80), confidence=0.80)  # area = 200 * 200 = 40000, center at (180, 120)
    face_center = DetectedFace(bounding_box=(150, 250, 250, 150), confidence=0.90) # area = 100 * 100 = 10000, center at (200, 200)

    faces = [face_small, face_large, face_center]

    # Strategy: largest
    primary_largest = detector.get_primary_face(faces, selection_strategy="largest")
    assert primary_largest == face_large

    # Strategy: highest_confidence
    primary_conf = detector.get_primary_face(faces, selection_strategy="highest_confidence")
    assert primary_conf == face_small

    # Strategy: center on a 400x400 image
    primary_center = detector.get_primary_face(faces, selection_strategy="center", image_shape=(400, 400))
    assert primary_center == face_center


# -------------------------------------------------------------------------
# M1: Face Cropper Tests
# -------------------------------------------------------------------------

def test_face_cropper_padding_and_saving():
    """Verify FaceCropper correctly extracts square crops, pads margins, and saves to file."""
    cropper = FaceCropper(default_padding_ratio=0.20, default_target_size=(512, 512))
    img = np.full((600, 600, 3), 128, dtype=np.uint8)
    face = DetectedFace(bounding_box=(100, 300, 300, 100))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "artifacts" / "face_crop.jpg"
        crop = cropper.crop_face(
            img,
            face,
            padding_ratio=0.25,
            target_size=(512, 512),
            save_path=save_path,
        )

        assert crop.shape == (512, 512, 3)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        # Verify saved file is a valid image
        loaded = cv2.imread(str(save_path))
        assert loaded is not None
        assert loaded.shape == (512, 512, 3)


def test_face_cropper_boundary_clamping():
    """Verify FaceCropper handles faces located at image edges without crashing."""
    cropper = FaceCropper()
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    # Face at the top-left boundary
    edge_face = DetectedFace(bounding_box=(0, 50, 50, 0))

    crop = cropper.crop_face(img, edge_face, padding_ratio=0.3, target_size=(256, 256))
    assert crop.shape == (256, 256, 3)


# -------------------------------------------------------------------------
# M1: Face Encoder & Similarity Metric Tests
# -------------------------------------------------------------------------

def test_face_encoder_similarity_metrics():
    """Verify cosine similarity and euclidean distance calculations."""
    encoder = FaceEncoder()

    # Identical unit vectors
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0, 0.0]
    assert math.isclose(encoder.cosine_similarity(v1, v2), 1.0, rel_tol=1e-5)
    assert math.isclose(encoder.euclidean_distance(v1, v2), 0.0, abs_tol=1e-5)
    assert encoder.is_match(v1, v2, similarity_threshold=0.60) is True

    # Orthogonal vectors
    v3 = [0.0, 1.0, 0.0, 0.0]
    assert math.isclose(encoder.cosine_similarity(v1, v3), 0.0, abs_tol=1e-5)
    assert math.isclose(encoder.euclidean_distance(v1, v3), math.sqrt(2.0), rel_tol=1e-5)
    assert encoder.is_match(v1, v3, similarity_threshold=0.60) is False

    # Close vectors (cos sim ~ 0.894)
    v4 = [0.8, 0.6, 0.0, 0.0]
    sim = encoder.cosine_similarity(v1, v4)
    assert math.isclose(sim, 0.80, rel_tol=1e-3)
    assert encoder.is_match(v1, v4, similarity_threshold=0.60) is True


def test_face_encoder_normalization():
    """Verify _normalize_vector produces unit vectors."""
    encoder = FaceEncoder()
    raw = np.array([3.0, 4.0, 0.0, 0.0])
    normalized = encoder._normalize_vector(raw)
    norm = np.linalg.norm(normalized)
    assert math.isclose(norm, 1.0, rel_tol=1e-6)
    assert math.isclose(normalized[0], 0.6, rel_tol=1e-6)
    assert math.isclose(normalized[1], 0.8, rel_tol=1e-6)


def test_face_encoder_raises_on_no_face():
    """Verify encode_face raises ValueError when no face exists in image."""
    encoder = FaceEncoder()
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No face detected"):
        encoder.encode_face(blank_img)


def test_face_detector_input_types():
    """Verify FaceDetector correctly parses numpy array, PIL Image, bytes, and file path."""
    detector = FaceDetector()
    img_np = np.full((150, 150, 3), 180, dtype=np.uint8)

    # 1. Numpy array
    rgb_from_np = detector._load_image_as_rgb(img_np)
    assert rgb_from_np.shape == (150, 150, 3)

    # 2. PIL Image
    pil_img = Image.fromarray(img_np)
    rgb_from_pil = detector._load_image_as_rgb(pil_img)
    assert rgb_from_pil.shape == (150, 150, 3)

    # 3. Bytes
    _, encoded_jpg = cv2.imencode(".jpg", img_np)
    bytes_data = encoded_jpg.tobytes()
    rgb_from_bytes = detector._load_image_as_rgb(bytes_data)
    assert rgb_from_bytes.shape == (150, 150, 3)

    # 4. File Path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), img_np)
        try:
            rgb_from_path = detector._load_image_as_rgb(tmp_path)
            assert rgb_from_path.shape == (150, 150, 3)
        finally:
            tmp_path.unlink(missing_ok=True)


def test_face_cropper_input_formats():
    """Verify FaceCropper works with PIL Image, numpy array, bytes, and file path."""
    cropper = FaceCropper()
    img_np = np.full((300, 300, 3), 150, dtype=np.uint8)
    face = DetectedFace(bounding_box=(50, 150, 150, 50))

    # PIL Image
    pil_img = Image.fromarray(img_np)
    crop_pil = cropper.crop_face(pil_img, face, target_size=(128, 128))
    assert crop_pil.shape == (128, 128, 3)

    # Bytes
    _, buf = cv2.imencode(".png", img_np)
    crop_bytes = cropper.crop_face(buf.tobytes(), face, target_size=(256, 256))
    assert crop_bytes.shape == (256, 256, 3)


def test_face_pipeline_integration_flow():
    """Verify end-to-end face processing flow produces valid FaceDetectionResult model."""
    detector = FaceDetector()
    cropper = FaceCropper()

    # Create synthetic face image
    face_img = create_synthetic_face_image(400, 400)

    # Provide a detected face
    face = DetectedFace(bounding_box=(50, 350, 350, 50), confidence=0.98)

    with tempfile.TemporaryDirectory() as tmpdir:
        crop_path = Path(tmpdir) / "face_crop.jpg"
        crop_np = cropper.crop_face(
            face_img,
            face,
            padding_ratio=0.15,
            target_size=(512, 512),
            save_path=crop_path,
        )

        # Build 128-d dummy unit vector embedding
        raw_emb = [1.0 / math.sqrt(128)] * 128
        norm_val = math.sqrt(sum(x ** 2 for x in raw_emb))
        assert math.isclose(norm_val, 1.0, rel_tol=1e-5)

        result = FaceDetectionResult(
            face_detected=True,
            bounding_box=face.bounding_box,
            embedding=raw_emb,
            face_crop_path=crop_path,
            confidence=face.confidence,
            detector_backend="face_recognition",
            faces_count=1,
            all_bounding_boxes=[face.bounding_box],
            image_shape=face_img.shape,
            processing_time_ms=45.2,
        )

        assert result.face_detected is True
        assert result.face_crop_path.exists()
        assert len(result.embedding) == 128
        assert result.image_shape == (400, 400, 3)

        # Validate JSON serialization
        json_data = result.model_dump_json()
        assert "face_detected" in json_data
        assert "bounding_box" in json_data
