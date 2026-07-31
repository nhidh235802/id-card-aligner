"""
vis_utils.py – Vẽ debug overlay lên ảnh để kiểm tra kết quả.
"""

import cv2
import numpy as np
from src.detector.base import DetectionResult

CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0, 255, 0),    # TL – xanh lá
    (255, 0, 0),    # TR – xanh dương
    (0, 0, 255),    # BR – đỏ
    (0, 255, 255),  # BL – vàng
]


def draw_detection(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Vẽ 4 góc, bounding box, angle và confidence lên ảnh."""
    vis = image.copy()
    corners = result.corners.astype(int)

    # Vẽ polygon thẻ
    cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

    # Vẽ từng góc
    for i, (pt, label, color) in enumerate(zip(corners, CORNER_LABELS, CORNER_COLORS)):
        cv2.circle(vis, tuple(pt), 6, color, -1)
        conf_str = ""
        if result.corner_confidences is not None:
            conf_str = f" ({result.corner_confidences[i]:.2f})"
        cv2.putText(vis, f"{label}{conf_str}", tuple(pt + [5, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Thông tin tổng hợp
    info = (
        f"conf={result.confidence:.2f}  "
        f"angle={result.angle_deg:.1f}°  "
        f"ratio={result.aspect_ratio:.3f}  "
        f"{'[OCCLUDED]' if result.is_occluded else ''}"
    )
    cv2.putText(vis, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis
