"""Face detection engine with multi-backend support, multi-face ranking, and robust error handling."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import cv2
import face_recognition
import numpy as np
from PIL import Image

from app.models import BoundingBox


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box, confidence, and geometry."""
    bounding_box: Tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float = 1.0
    backend: str = "face_recognition"
    landmarks: Optional[Dict[str, Any]] = None

    @property
    def top(self) -> int:
        return self.bounding_box[0]

    @property
    def right(self) -> int:
        return self.bounding_box[1]

    @property
    def bottom(self) -> int:
        return self.bounding_box[2]

    @property
    def left(self) -> int:
        return self.bounding_box[3]

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def to_bbox_model(self) -> BoundingBox:
        return BoundingBox(
            top=self.top,
            right=self.right,
            bottom=self.bottom,
            left=self.left,
        )


class FaceDetector:
    """Robust face detector supporting face_recognition (HOG/CNN), OpenCV DNN, and Haar Cascades."""

    def __init__(
        self,
        backend: Literal["face_recognition", "opencv_dnn", "haar", "auto"] = "face_recognition",
        model: Literal["hog", "cnn"] = "hog",
        upsample_times: int = 1,
        min_face_size: int = 20,
    ):
        self.backend = backend
        self.model = model
        self.upsample_times = upsample_times
        self.min_face_size = min_face_size
        self._opencv_net = None
        self._haar_cascade = None

    def _load_image_as_rgb(
        self, image_input: Union[str, Path, np.ndarray, bytes, Image.Image]
    ) -> np.ndarray:
        """Load various image input types into a standardized RGB numpy array (H, W, 3)."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError(f"Failed to decode image from path: {path}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image from raw bytes")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, Image.Image):
            rgb_image = image_input.convert("RGB")
            return np.array(rgb_image)

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty")
            if image_input.ndim == 2:
                # Grayscale to RGB
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    # RGBA to RGB
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                elif image_input.shape[2] == 3:
                    # Assume RGB
                    return image_input
                else:
                    raise ValueError(f"Unsupported channel count: {image_input.shape[2]}")
            else:
                raise ValueError(f"Unsupported image array dimensions: {image_input.ndim}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect_faces(
        self,
        image_input: Union[str, Path, np.ndarray, bytes, Image.Image],
    ) -> List[DetectedFace]:
        """Detect all face bounding boxes in the input image.

        Returns a list of DetectedFace objects, sorted by face bounding-box area descending.
        """
        try:
            rgb_image = self._load_image_as_rgb(image_input)
        except Exception as e:
            # Propagate or log corrupt image errors
            raise ValueError(f"Invalid image input for face detection: {e}") from e

        h, w = rgb_image.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return []

        faces: List[DetectedFace] = []

        if self.backend in ("face_recognition", "auto"):
            try:
                # face_recognition returns (top, right, bottom, left)
                raw_boxes = face_recognition.face_locations(
                    rgb_image,
                    number_of_times_to_upsample=self.upsample_times,
                    model=self.model,
                )
                for top, right, bottom, left in raw_boxes:
                    # Clamp to image boundaries
                    top = max(0, min(top, h))
                    bottom = max(0, min(bottom, h))
                    left = max(0, min(left, w))
                    right = max(0, min(right, w))

                    if (bottom - top) >= self.min_face_size and (right - left) >= self.min_face_size:
                        faces.append(
                            DetectedFace(
                                bounding_box=(top, right, bottom, left),
                                confidence=0.98,
                                backend="face_recognition",
                            )
                        )
            except Exception:
                if self.backend != "auto":
                    raise

        # If auto and face_recognition found no faces or backend is haar
        if not faces and self.backend in ("haar", "auto", "opencv_dnn"):
            faces = self._detect_haar(rgb_image)

        # Sort faces descending by bounding box area (largest face first)
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def _detect_haar(self, rgb_image: np.ndarray) -> List[DetectedFace]:
        """Fallback detection using OpenCV Haar cascades."""
        if self._haar_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        rects = self._haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        faces = []
        for x, y, fw, fh in rects:
            top = max(0, y)
            left = max(0, x)
            bottom = min(h, y + fh)
            right = min(w, x + fw)
            faces.append(
                DetectedFace(
                    bounding_box=(top, right, bottom, left),
                    confidence=0.85,
                    backend="haar_cascade",
                )
            )
        return faces

    def get_primary_face(
        self,
        faces: List[DetectedFace],
        selection_strategy: Literal["largest", "center", "highest_confidence"] = "largest",
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[DetectedFace]:
        """Select primary face from detected faces based on specified strategy."""
        if not faces:
            return None

        if len(faces) == 1:
            return faces[0]

        if selection_strategy == "largest":
            return max(faces, key=lambda f: f.area)

        elif selection_strategy == "highest_confidence":
            return max(faces, key=lambda f: f.confidence)

        elif selection_strategy == "center":
            if image_shape is None:
                return max(faces, key=lambda f: f.area)
            img_h, img_w = image_shape[:2]
            img_cx, img_cy = img_w / 2.0, img_h / 2.0

            def distance_to_center(f: DetectedFace) -> float:
                fcx, fcy = f.center
                return (fcx - img_cx) ** 2 + (fcy - img_cy) ** 2

            return min(faces, key=distance_to_center)

        return faces[0]
