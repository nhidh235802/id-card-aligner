"""
OBBDetector – dùng YOLO-OBB để detect oriented bounding box của thẻ.

Từ OBB (cx, cy, w, h, angle) → suy ra 4 góc và angle.
Không cho corner confidence riêng lẻ → occlusion handling bị hạn chế hơn Pose.
"""

import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult
from src.utils.corner_utils import obb_to_corners, order_corners, compute_angle, compute_aspect_ratio


class OBBDetector(BaseDetector):
    """YOLO-OBB based card detector."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.imgsz = config.get("imgsz", 640)

    def load_model(self):
        self.model = YOLO(self.config["weights"])
        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results or results[0].obb is None or len(results[0].obb) == 0:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=False,
            )

        obb = results[0].obb
        # Lấy detection có confidence cao nhất
        best_idx = int(obb.conf.argmax())
        box_conf = float(obb.conf[best_idx].cpu())

        # xywhr: (cx, cy, w, h, angle_rad)
        xywhr = obb.xywhr[best_idx].cpu().numpy()
        corners = obb_to_corners(xywhr)          # (4, 2)
        corners = order_corners(corners)

        angle = compute_angle(corners)
        aspect = compute_aspect_ratio(corners)

        return DetectionResult(
            corners=corners.astype(np.float32),
            confidence=box_conf,
            angle_deg=angle,
            aspect_ratio=aspect,
            is_occluded=False,   # OBB không estimate occlusion per-corner
            corner_confidences=None,
        )
