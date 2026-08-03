"""
PoseDetector – dùng YOLO-Pose (keypoint) để detect 4 góc thẻ CCCD.

Keypoint order: [TL, TR, BR, BL]  (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
"""

import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult
from src.utils.corner_utils import order_corners, compute_angle, compute_aspect_ratio
from src.utils.occlusion_utils import handle_missing_corners
from src.utils.subpixel_utils import refine_corners_subpixel


class PoseDetector(BaseDetector):
    """
    YOLO-Pose based card corner detector.

    Args:
        config (dict): xem configs/pose_detector.yaml
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.imgsz = config.get("imgsz", 640)
        self.occlusion_min_conf = config.get("occlusion_min_conf", 0.3)
        self.use_subpixel = config.get("use_subpixel", True)

    def load_model(self):
        weights = self.config["weights"]
        self.model = YOLO(weights)
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

        if not results or results[0].keypoints is None:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        kpts = results[0].keypoints  # Keypoints object
        if kpts is None or len(kpts.xy) == 0:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        xy = kpts.xy[0].cpu().numpy()          # Shape (N, 2)
        conf_per_kpt = kpts.conf[0].cpu().numpy() if kpts.conf is not None else None
        box_conf = float(results[0].boxes.conf[0].cpu()) if len(results[0].boxes) > 0 else 0.0

        # An toàn: Nếu số lượng keypoint khác 4 (do xài pretrained COCO 17 keypoints)
        # thì coi như chưa detect được thẻ CCCD
        if len(xy) != 4:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        # ── Occlusion handling: góc nào conf thấp → ước tính lại ──────────
        is_occluded = bool(np.any(conf_per_kpt < self.occlusion_min_conf))
        if is_occluded:
            xy = handle_missing_corners(xy, conf_per_kpt, self.occlusion_min_conf)

        # ── Sắp xếp góc đúng thứ tự [TL, TR, BR, BL] ──────────────────────
        corners = order_corners(xy)

        # ── Sub-pixel refinement ────────────────────────────────────────────
        if self.use_subpixel:
            corners = refine_corners_subpixel(image, corners)

        angle = compute_angle(corners)
        aspect = compute_aspect_ratio(corners)

        return DetectionResult(
            corners=corners.astype(np.float32),
            confidence=box_conf,
            angle_deg=angle,
            aspect_ratio=aspect,
            is_occluded=is_occluded,
            corner_confidences=conf_per_kpt,
        )
