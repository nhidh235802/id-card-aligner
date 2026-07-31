"""
subpixel_utils.py – Làm mịn vị trí góc thẻ ở mức sub-pixel.

Dùng OpenCV cornerSubPix để tăng độ chính xác tọa độ góc
từ ~1px → ~0.1px, giúp perspective transform chính xác hơn.

Quan trọng nhất khi:
- Thẻ được chụp gần (độ phân giải cao)
- OCR cần crop sát cạnh mà không mất pixel
"""

import cv2
import numpy as np


def refine_corners_subpixel(
    image: np.ndarray,
    corners: np.ndarray,
    win_size: int = 5,
    max_iter: int = 30,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    Tinh chỉnh tọa độ 4 góc ở mức sub-pixel.

    Args:
        image    : BGR image gốc (H, W, 3)
        corners  : (4, 2) float32 – góc từ detector
        win_size : kích thước cửa sổ tìm kiếm (pixel)
        max_iter : số vòng lặp tối đa
        epsilon  : ngưỡng hội tụ

    Returns:
        refined_corners : (4, 2) float32
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon,
    )

    corners_cv = corners.reshape(-1, 1, 2).astype(np.float32)

    try:
        refined = cv2.cornerSubPix(
            gray,
            corners_cv,
            winSize=(win_size, win_size),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        return refined.reshape(4, 2).astype(np.float32)
    except cv2.error:
        # Nếu góc nằm ngoài ảnh → trả về nguyên bản
        return corners
