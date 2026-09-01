"""Face image cropping, aspect-ratio preserving padding, and artifact persistence."""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from app.face.detector import DetectedFace, FaceDetector


class FaceCropper:
    """Crops detected facial regions with configurable padding, square aspect ratio, and disk persistence."""

    def __init__(self, default_padding_ratio: float = 0.20, default_target_size: Tuple[int, int] = (512, 512)):
        self.default_padding_ratio = default_padding_ratio
        self.default_target_size = default_target_size
        self.detector = FaceDetector()

    def crop_face(
        self,
        image_input: Union[str, Path, np.ndarray, bytes, Image.Image],
        face: DetectedFace,
        padding_ratio: Optional[float] = None,
        target_size: Optional[Tuple[int, int]] = (512, 512),
        save_path: Optional[Union[str, Path]] = None,
        quality: int = 95,
    ) -> np.ndarray:
        """Crop the face region from image, apply padding, resize, and optionally save to disk.

        Args:
            image_input: Source image (Path, str, numpy array, bytes, or PIL Image).
            face: DetectedFace containing bounding box coordinates.
            padding_ratio: Fractional padding around face box (default 0.20 = 20% on each side).
            target_size: Optional (width, height) to resize crop (default (512, 512)).
            save_path: Optional file path to persist cropped JPEG (e.g., 'artifacts/face_crop.jpg').
            quality: JPEG quality if saving (1-100).

        Returns:
            np.ndarray: Cropped RGB image array.
        """
        rgb_image = self.detector._load_image_as_rgb(image_input)
        img_h, img_w = rgb_image.shape[:2]

        pad_ratio = padding_ratio if padding_ratio is not None else self.default_padding_ratio
        top, right, bottom, left = face.bounding_box

        box_w = right - left
        box_h = bottom - top

        # Calculate square dimension with padding
        max_dim = max(box_w, box_h)
        pad_px = int(max_dim * pad_ratio)

        # Center of bounding box
        cx = left + box_w / 2.0
        cy = top + box_h / 2.0

        half_side = (max_dim / 2.0) + pad_px

        crop_top = int(round(cy - half_side))
        crop_bottom = int(round(cy + half_side))
        crop_left = int(round(cx - half_side))
        crop_right = int(round(cx + half_side))

        # Handle out-of-boundary coordinates by padding the image if necessary
        pad_top = max(0, -crop_top)
        pad_bottom = max(0, crop_bottom - img_h)
        pad_left = max(0, -crop_left)
        pad_right = max(0, crop_right - img_w)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_img = cv2.copyMakeBorder(
                rgb_image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            adj_top = crop_top + pad_top
            adj_bottom = crop_bottom + pad_top
            adj_left = crop_left + pad_left
            adj_right = crop_right + pad_left
            cropped = padded_img[adj_top:adj_bottom, adj_left:adj_right]
        else:
            cropped = rgb_image[crop_top:crop_bottom, crop_left:crop_right]

        if cropped.size == 0:
            # Fallback to direct bbox crop
            cropped = rgb_image[top:bottom, left:right]

        # Resize if target_size is specified
        if target_size is not None:
            target_w, target_h = target_size
            curr_h, curr_w = cropped.shape[:2]
            interpolation = cv2.INTER_AREA if (curr_w > target_w or curr_h > target_h) else cv2.INTER_LANCZOS4
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=interpolation)

        # Persist to disk if save_path specified
        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bgr_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(out_path),
                bgr_crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )

        return cropped
