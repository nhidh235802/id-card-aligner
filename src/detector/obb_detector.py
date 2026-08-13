import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult
from src.utils.corner_utils import obb_to_corners, order_corners, compute_angle, compute_aspect_ratio


class OBBDetector(BaseDetector):
    """Bộ phát hiện góc thẻ căn cước sử dụng mô hình YOLO-OBB (Oriented Bounding Box)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.imgsz = config.get("imgsz", 640)

    def load_model(self):
        """Khởi tạo mô hình YOLO-OBB từ file trọng số (weights)."""
        self.model = YOLO(self.config["weights"])
        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Dự đoán bounding box có hướng xoay (OBB) và trích xuất tọa độ 4 góc thẻ."""
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")

        # ── Khối 1: Khởi chạy suy luận dự đoán mô hình YOLO-OBB ──
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        # Trả về kết quả rỗng nếu không phát hiện được OBB thẻ nào
        if not results or results[0].obb is None or len(results[0].obb) == 0:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=False,
            )

        # ── Khối 2: Trích xuất hộp OBB có điểm độ tin cậy confidence cao nhất ──
        obb = results[0].obb
        best_idx = int(obb.conf.argmax())
        box_conf = float(obb.conf[best_idx].cpu())

        # ── Khối 3: Bóc tách và sắp xếp 4 đỉnh theo thứ tự chuẩn [TL, TR, BR, BL] ──
        pts = obb.xyxyxyxy[best_idx].cpu().numpy()
        corners = pts[[2, 1, 0, 3]].astype(np.float32)

        # Tính toán góc xoay và tỷ lệ chiều rộng/chiều cao
        angle = compute_angle(corners)
        aspect = compute_aspect_ratio(corners)

        return DetectionResult(
            corners=corners.astype(np.float32),
            confidence=box_conf,
            angle_deg=angle,
            aspect_ratio=aspect,
            is_occluded=False,
            corner_confidences=None,
        )
