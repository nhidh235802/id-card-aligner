from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """Cấu trúc dữ liệu kết quả phát hiện 4 góc thẻ thống nhất cho tất cả mô hình detector."""
    corners: np.ndarray                        # Mảng (4, 2) dtype float32 lưu tọa độ [TL, TR, BR, BL]
    confidence: float = 0.0                    # Độ tin cậy tổng thể [0.0, 1.0]
    angle_deg: float = 0.0                     # Góc nghiêng ước tính (độ)
    aspect_ratio: float = 1.586                # Tỷ lệ r/c (width/height) của thẻ
    is_occluded: bool = False                  # Cờ đánh dấu có bị che khuất góc hay không
    corner_confidences: Optional[np.ndarray] = None   # Mảng (4,) lưu độ tin cậy riêng của từng góc
    extra_heads: Optional[dict] = None         # Dữ liệu bổ sung cho mô hình đa nhiệm trong tương lai


class BaseDetector(ABC):
    """Lớp cơ sở trừu tượng (Abstract Base Class) định nghĩa giao diện chung cho mọi bộ phát hiện góc thẻ."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Nhận đầu vào ảnh BGR (H, W, 3) và trả về đối tượng DetectionResult chứa thông tin 4 góc."""
        ...

    @abstractmethod
    def load_model(self):
        """Khởi tạo mô hình và tải trọng số huấn luyện (weights)."""
        ...

    def warmup(self, iterations: int = 3):
        """Chạy thử suy luận trên mảng ma trận trống để làm nóng mô hình trên GPU/CUDA."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
