import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult
from src.utils.corner_utils import order_corners, compute_angle, compute_aspect_ratio
from src.utils.occlusion_utils import handle_missing_corners
from src.utils.subpixel_utils import refine_corners_subpixel


class PoseDetector(BaseDetector):
    """Bộ phát hiện góc thẻ căn cước bằng mô hình điểm mốc YOLO-Pose (Keypoint Detection)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.imgsz = config.get("imgsz", 640)
        self.occlusion_min_conf = config.get("occlusion_min_conf", 0.3)
        self.use_subpixel = config.get("use_subpixel", True)

    def load_model(self):
        """Khởi tạo mô hình YOLO-Pose từ file trọng số (weights)."""
        weights = self.config["weights"]
        self.model = YOLO(weights)
        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Dự đoán 4 keypoints góc thẻ theo đúng thứ tự không gian ngữ nghĩa [TL, TR, BR, BL]."""
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")

        # ── Khối 1: Chạy mô hình YOLO-Pose để dự đoán vị trí keypoints ──
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

        kpts = results[0].keypoints
        if kpts is None or len(kpts.xy) == 0:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        # ── Khối 2: Trích xuất tọa độ pixel và độ tin cậy của từng keypoint góc ──
        xy = kpts.xy[0].cpu().numpy()          # Shape (N, 2)
        conf_per_kpt = kpts.conf[0].cpu().numpy() if kpts.conf is not None else None
        box_conf = float(results[0].boxes.conf[0].cpu()) if len(results[0].boxes) > 0 else 0.0

        # Kiểm tra điều kiện số lượng keypoints đủ 4 góc thẻ
        if len(xy) != 4:
            return DetectionResult(
                corners=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                is_occluded=True,
            )

        # ── Khối 3: Xử lý che khuất (Occlusion Handling) cho điểm góc có độ tin cậy thấp ──
        is_occluded = bool(np.any(conf_per_kpt < self.occlusion_min_conf))
        if is_occluded:
            xy = handle_missing_corners(xy, conf_per_kpt, self.occlusion_min_conf)

        # Giữ nguyên thứ tự không gian học được từ nhãn Pose [TL, TR, BR, BL]
        corners = xy.astype(np.float32)

        # ── Khối 4: Tinh chỉnh vị trí góc mức dưới pixel (Sub-pixel Refinement) ──
        if self.use_subpixel:
            corners = refine_corners_subpixel(image, corners)

        # Tính toán các giá trị hình học góc xoay và tỷ lệ
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
