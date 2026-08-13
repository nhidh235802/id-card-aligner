import cv2
import numpy as np


def refine_corners_subpixel(
    image: np.ndarray,
    corners: np.ndarray,
    win_size: int = 5,
    max_iter: int = 30,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Tinh chỉnh tọa độ 4 góc thẻ ở độ chính xác dưới mức pixel (sub-pixel) bằng thuật toán OpenCV cornerSubPix."""
    # Chuyển ảnh BGR nguồn sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Khối 1: Định nghĩa tiêu chí dừng hội tụ (vòng lặp tối đa hoặc ngưỡng sai số epsilon) ──
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon,
    )

    corners_cv = corners.reshape(-1, 1, 2).astype(np.float32)

    # ── Khối 2: Thực thi thuật toán cornerSubPix tinh chỉnh tọa độ góc ──
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
        # Nếu có lỗi ngoại lệ hoặc góc nằm ngoài phạm vi ảnh -> trả về tọa độ ban đầu
        return corners
