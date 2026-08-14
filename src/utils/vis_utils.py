import cv2
import numpy as np
from src.detector.base import DetectionResult

# Nhãn góc và màu sắc tương ứng [TL, TR, BR, BL]
CORNER_LABELS  = ["TL", "TR", "BR", "BL"]
CORNER_COLORS  = [
    (0,   255, 0),    # TL – Xanh lá
    (255, 0,   0),    # TR – Xanh dương
    (0,   0,   255),  # BR – Đỏ
    (0,   255, 255),  # BL – Vàng
]

# Màu dùng cho visualization kiểu YOLO (giống ảnh mentor)
COLOR_BOX      = (0,   200, 0)    # Viền hộp bao – xanh lá
COLOR_KPT      = (0,   0,   200)  # Chấm keypoint – đỏ đậm
COLOR_LABEL_BG = (0,   0,   0)    # Nền chữ label – đen (nếu cần)
COLOR_LABEL_FG = (0,   255, 0)    # Chữ label class – xanh lá sáng


def draw_detection_result(image: np.ndarray, result: DetectionResult,
                          show_corner_labels: bool = False) -> np.ndarray:
    """Vẽ kết quả detection theo đúng format YOLO-pose như ảnh kết quả của mentor.

    Layout:
        - Góc trên-trái: "<class_name>  <confidence>" (chữ lớn, màu xanh lá)
        - Hộp bao (bbox_xyxy nếu có, else bounding rect của keypoints): viền xanh lá
        - 4 chấm đỏ tại vị trí keypoints 4 góc thẻ
        - Đa giác nối 4 góc (polyline) xanh lá mảnh
    """
    vis = image.copy()
    corners = result.corners.astype(np.int32)

    # ── 1. Vẽ bounding box từ model (bbox_xyxy) ──────────────────────────────
    if result.bbox_xyxy is not None:
        x1, y1, x2, y2 = result.bbox_xyxy.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BOX, 2)
    else:
        # Fallback: tính bounding rect từ keypoints nếu không có bbox
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BOX, 2)

    # ── 2. Vẽ đa giác nối 4 góc (outline thẻ) ───────────────────────────────
    cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, COLOR_BOX, 1)

    # ── 3. Vẽ 4 chấm keypoint tại góc thẻ ───────────────────────────────────
    for i, pt in enumerate(corners):
        cv2.circle(vis, tuple(pt), 7, COLOR_KPT, -1)          # Chấm đỏ đặc
        cv2.circle(vis, tuple(pt), 9, (255, 255, 255), 1)     # Viền trắng mỏng
        if show_corner_labels:
            conf_str = ""
            if result.corner_confidences is not None:
                conf_str = f" {result.corner_confidences[i]:.2f}"
            cv2.putText(vis, f"{CORNER_LABELS[i]}{conf_str}",
                        (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORNER_COLORS[i], 1)

    # ── 4. Vẽ label "<class_name>  <conf>" lớn ở góc trên-trái bbox ─────────
    label_text = f"{result.class_name}  {result.confidence:.2f}"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = _auto_font_scale(vis.shape[1])
    thickness   = max(1, int(font_scale * 2.2))

    # Vị trí: trên bbox hoặc đầu ảnh nếu bbox sát mép
    label_x = x1
    label_y = max(y1 - 8, int(font_scale * 28) + 4)

    # Shadow đen để dễ đọc trên mọi nền
    cv2.putText(vis, label_text, (label_x + 2, label_y + 2),
                font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(vis, label_text, (label_x, label_y),
                font, font_scale, COLOR_LABEL_FG, thickness)

    return vis


def draw_detection(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Wrapper tương thích ngược – gọi draw_detection_result() kèm corner labels.

    Giữ nguyên để không phá vỡ code cũ đang import hàm này.
    """
    return draw_detection_result(image, result, show_corner_labels=True)


def draw_detection_verbose(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Vẽ đầy đủ thông tin debug: class, conf, angle, ratio, occlusion, corner conf."""
    vis = draw_detection_result(image, result, show_corner_labels=True)

    # Thông tin phụ góc dưới-trái
    info_lines = [
        f"class_id={result.class_id}  angle={result.angle_deg:.1f}deg",
        f"ratio={result.aspect_ratio:.3f}  {'[OCCLUDED]' if result.is_occluded else 'OK'}",
    ]
    h = vis.shape[0]
    for i, line in enumerate(reversed(info_lines)):
        y = h - 10 - i * 22
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
    return vis


# ── Hàm tiện ích nội bộ ────────────────────────────────────────────────────────

def _auto_font_scale(img_width: int) -> float:
    """Tự động chọn cỡ chữ tỷ lệ với chiều rộng ảnh (giống style YOLO official)."""
    return max(0.6, img_width / 900)
