"""
ClassicalDetector – OpenCV-based card corner detection (no ML).

Pipeline:
    BGR → Gray → Blur → Canny → findContours
    → approxPolyDP (quadrilateral) → 4 corners

Dùng để:
  1. Baseline so sánh với YOLO methods
  2. Fallback khi model không có
"""

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from src.utils.corner_utils import order_corners, compute_angle, compute_aspect_ratio


class ClassicalDetector(BaseDetector):
    """OpenCV contour-based card detector (no model weights needed)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.canny_low = config.get("canny_low", 50)
        self.canny_high = config.get("canny_high", 150)
        self.blur_ksize = config.get("blur_ksize", 5)
        self.min_area_ratio = config.get("min_area_ratio", 0.1)  # 10% of image
        self.poly_epsilon = config.get("poly_epsilon", 0.02)

    def load_model(self):
        # Không cần model weights
        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        min_area = h * w * self.min_area_ratio

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilate để nối các cạnh bị đứt
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_corners = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.poly_epsilon * peri, True)
            if len(approx) == 4 and area > best_area:
                best_corners = approx.reshape(4, 2).astype(np.float32)
                best_area = area

        if best_corners is None:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        corners = order_corners(best_corners)
        angle = compute_angle(corners)
        aspect = compute_aspect_ratio(corners)
        confidence = min(best_area / (h * w), 1.0)

        return DetectionResult(
            corners=corners.astype(np.float32),
            confidence=float(confidence),
            angle_deg=angle,
            aspect_ratio=aspect,
            is_occluded=False,
        )
