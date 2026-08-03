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

    def _find_best_quad(self, edge_map: np.ndarray, min_area: float):
        """Tìm contour 4 cạnh lớn nhất từ edge map."""
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_corners = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.poly_epsilon * peri, True)
            if len(approx) == 4 and area > best_area:
                best_corners = approx.reshape(4, 2).astype(np.float32)
                best_area = area
        return best_corners, best_area

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        min_area = h * w * self.min_area_ratio

        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # ── Chiến lược 1: Canny edge ─────────────────────────────────────────
        edges  = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=2)
        best_corners, best_area = self._find_best_quad(edges, min_area)

        # ── Chiến lược 2: Otsu threshold (khi nền trùng màu thẻ) ─────────────
        # Canny thất bại khi gradient giữa thẻ và nền quá yếu.
        # Otsu phân ngưỡng toàn cục vẫn tách được blob thẻ khỏi nền.
        if best_corners is None:
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Thử cả 2 chiều: sáng-trên-tối và tối-trên-sáng
            for thresh_img in [otsu, cv2.bitwise_not(otsu)]:
                thresh_d = cv2.dilate(thresh_img, kernel, iterations=1)
                thresh_e = cv2.erode(thresh_d, kernel, iterations=1)
                corners, area = self._find_best_quad(thresh_e, min_area)
                if corners is not None and area > best_area:
                    best_corners, best_area = corners, area

        if best_corners is None:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        corners = order_corners(best_corners)
        angle   = compute_angle(corners)
        aspect  = compute_aspect_ratio(corners)

        # Confidence dựa trên diện tích tương đối của thẻ so với ảnh
        # Thẻ chiếm khoảng 30-65% canvas → normalize về [0, 1]
        confidence = min(best_area / (h * w * 0.65), 1.0)

        return DetectionResult(
            corners=corners.astype(np.float32),
            confidence=float(confidence),
            angle_deg=angle,
            aspect_ratio=aspect,
            is_occluded=False,
        )
