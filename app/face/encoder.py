"""Face feature encoder for extracting normalized 128-d / 512-d facial embeddings and metric calculations."""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
import cv2
import face_recognition
import numpy as np
from PIL import Image

from app.face.detector import DetectedFace, FaceDetector


class FaceEncoder:
    """Extracts normalized 128-d face embeddings and provides similarity metric calculations."""

    def __init__(
        self,
        dimension: int = 128,
        num_jitters: int = 1,
        model: str = "small",  # "small" or "large" in face_recognition
    ):
        self.dimension = dimension
        self.num_jitters = num_jitters
        self.model = model
        self.detector = FaceDetector()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0 or np.isnan(norm):
            return vector
        return vector / norm

    def encode_face(
        self,
        image_input: Union[str, Path, np.ndarray, bytes, Image.Image],
        face: Optional[DetectedFace] = None,
        num_jitters: Optional[int] = None,
    ) -> List[float]:
        """Extract a normalized 128-d face embedding from an image.

        If `face` is provided, computes embedding for that bounding box.
        If `face` is None, detects primary face first.
        Raises ValueError if no face is detected or image is invalid.
        """
        rgb_image = self.detector._load_image_as_rgb(image_input)
        jitters = num_jitters if num_jitters is not None else self.num_jitters

        if face is not None:
            known_face_locations = [face.bounding_box]
        else:
            detected_faces = self.detector.detect_faces(rgb_image)
            if not detected_faces:
                raise ValueError("Cannot extract face embedding: No face detected in the image.")
            primary_face = self.detector.get_primary_face(detected_faces, image_shape=rgb_image.shape[:2])
            known_face_locations = [primary_face.bounding_box]

        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=known_face_locations,
            num_jitters=jitters,
            model=self.model,
        )

        if not encodings or len(encodings) == 0:
            raise ValueError("Face recognition failed to compute embedding features for the face region.")

        raw_vec = np.array(encodings[0], dtype=np.float64)
        normalized_vec = self._normalize_vector(raw_vec)
        return normalized_vec.tolist()

    @staticmethod
    def cosine_similarity(emb1: Sequence[float], emb2: Sequence[float]) -> float:
        """Calculate cosine similarity between two embedding vectors.

        Returns float between -1.0 and 1.0 (clamped between 0.0 and 1.0 for normalized face embeddings).
        """
        v1 = np.array(emb1, dtype=np.float64)
        v2 = np.array(emb2, dtype=np.float64)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot = np.dot(v1, v2)
        sim = float(dot / (norm1 * norm2))
        return float(np.clip(sim, 0.0, 1.0))

    @staticmethod
    def euclidean_distance(emb1: Sequence[float], emb2: Sequence[float]) -> float:
        r"""Calculate Euclidean distance between two embedding vectors.

        For unit-normalized vectors, Euclidean distance $d = \sqrt{2 - 2 \cdot \text{cos\_sim}}$.
        """
        v1 = np.array(emb1, dtype=np.float64)
        v2 = np.array(emb2, dtype=np.float64)
        return float(np.linalg.norm(v1 - v2))

    def is_match(
        self,
        emb1: Sequence[float],
        emb2: Sequence[float],
        similarity_threshold: float = 0.60,
    ) -> bool:
        """Check if two face embeddings belong to the same person based on cosine similarity."""
        sim = self.cosine_similarity(emb1, emb2)
        return sim >= similarity_threshold
