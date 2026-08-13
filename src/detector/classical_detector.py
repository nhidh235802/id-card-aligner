import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from src.utils.corner_utils import order_corners, compute_angle, compute_aspect_ratio


class ClassicalDetector(BaseDetector):
    """Bộ phát hiện góc thẻ bằng kỹ thuật xử lý ảnh OpenCV truyền thống (không sử dụng Deep Learning)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.canny_low = config.get("canny_low", 50)
        self.canny_high = config.get("canny_high", 150)
        self.blur_ksize = config.get("blur_ksize", 5)
        self.min_area_ratio = config.get("min_area_ratio", 0.1)  # Ngưỡng diện tích 10% ảnh
        self.poly_epsilon = config.get("poly_epsilon", 0.02)

    def load_model(self):
        """Phương pháp truyền thống không cần tải file trọng số."""
        return self

    def _find_best_quad(self, edge_map: np.ndarray, min_area: float):
        """Tìm ứng viên contour có đúng 4 cạnh (quadrilateral) có diện tích lớn nhất."""
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
        """Thực hiện quy trình phát hiện 4 góc thẻ bằng hai chiến lược Canny và Otsu."""
        h, w = image.shape[:2]
        min_area = h * w * self.min_area_ratio

        # Tiền xử lý ảnh xám và làm mờ giảm nhiễu bằng Gaussian Blur
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # ── Chiến lược 1: Phát hiện đường viền bằng thuật toán Canny edge ──
        edges  = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges  = cv2.dilate(edges, kernel, iterations=2)
        best_corners, best_area = self._find_best_quad(edges, min_area)

        # ── Chiến lược 2: Phương pháp phân ngưỡng Otsu threshold (Fallback nếu Canny thất bại) ──
        if best_corners is None:
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Thử cả hai chiều nhị phân: tương phản sáng/tối và tối/sáng
            for thresh_img in [otsu, cv2.bitwise_not(otsu)]:
                thresh_d = cv2.dilate(thresh_img, kernel, iterations=1)
                thresh_e = cv2.erode(thresh_d, kernel, iterations=1)
                corners, area = self._find_best_quad(thresh_e, min_area)
                if corners is not None and area > best_area:
                    best_corners, best_area = corners, area

        # Nếu không tìm thấy contour 4 cạnh thỏa mãn -> Trả về kết quả rỗng
        if best_corners is None:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        # ── Khối chuẩn hóa thứ tự điểm góc và tính chỉ số hình học ──
        corners = order_corners(best_corners)
        angle   = compute_angle(corners)
        aspect  = compute_aspect_ratio(corners)

        # Tính toán độ tin cậy dựa trên tỷ lệ diện tích khung thẻ so với tổng diện tích ảnh
        confidence = min(best_area / (h * w * 0.65), 1.0)

        return DetectionResult(
            corners=corners.astype(np.float32),
            confidence=float(confidence),
            angle_deg=angle,
            aspect_ratio=aspect,
            is_occluded=False,
        )
