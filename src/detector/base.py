"""
BaseDetector – abstract interface for all corner detectors.

Thiết kế interface chung giúp:
- Swap dễ dàng giữa PoseDetector / OBBDetector / ClassicalDetector
- Tích hợp Multi-task model sau này chỉ cần kế thừa lớp này
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """
    Chuẩn đầu ra thống nhất cho mọi detector.

    corners : (4, 2) float32 – tọa độ [TL, TR, BR, BL] theo pixel
    confidence : float – độ tin cậy tổng thể [0, 1]
    angle_deg : float – góc xoay ước tính (độ), dương = nghiêng phải
    aspect_ratio : float – tỷ lệ w/h thẻ (chuẩn CCCD ≈ 1.586)
    is_occluded : bool – True nếu ít nhất 1 góc bị che / không nhìn thấy
    corner_confidences : (4,) float – confidence riêng từng góc (phục vụ
                          occlusion handling & sub-pixel refinement)

    NOTE: Khi tích hợp Multi-task model, chỉ cần trả thêm trường
          extra_heads (dict) mà không phá vỡ code hiện tại.
    """
    corners: np.ndarray                        # shape (4, 2), dtype float32
    confidence: float = 0.0
    angle_deg: float = 0.0
    aspect_ratio: float = 1.586                # ID card default
    is_occluded: bool = False
    corner_confidences: Optional[np.ndarray] = None   # shape (4,)
    extra_heads: Optional[dict] = None         # reserved for multi-task


class BaseDetector(ABC):
    """Abstract base class – mọi detector đều implement interface này."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Nhận ảnh BGR (H, W, 3) → trả DetectionResult.
        Đây là method duy nhất pipeline cần gọi.
        """
        ...

    @abstractmethod
    def load_model(self):
        """Load weights / khởi tạo model."""
        ...

    def warmup(self, iterations: int = 3):
        """Chạy dummy inference để JIT/CUDA warm-up (override nếu cần)."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
