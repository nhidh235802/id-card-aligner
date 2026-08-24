import numpy as np

# Tỷ lệ khung hình thẻ ID-1 chuẩn (Width / Height)
CCCD_ASPECT_RATIO = 85.6 / 54.0  # ≈ 1.585185


def handle_missing_corners(
    corners: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.4,
) -> tuple[np.ndarray, bool]:
    """Khôi phục điểm góc bị thiếu dựa trên 3 điểm có confidence cao.

    Args:
        corners: Mảng (4, 2) tọa độ [TL, TR, BR, BL]
        confidences: Mảng (4,) độ tin cậy của từng keypoint [0.0, 1.0]
        threshold: Ngưỡng confidence tối thiểu để tin cậy một keypoint (mặc định 0.4)

    Returns:
        tuple (refined_corners: np.ndarray, is_valid: bool)
        - is_valid = True: Đủ điều kiện khôi phục (0 hoặc 1 góc bị thiếu)
        - is_valid = False: Quá 1 góc bị thiếu (>= 2 góc kém), từ chối warp để tránh méo ảnh.
    """
    pts = corners.copy().astype(np.float32)

    if confidences is None:
        return pts, True

    missing = confidences < threshold
    n_missing = int(missing.sum())

    # Trường hợp 0: Cả 4 góc đều tin cậy -> Giữ nguyên
    if n_missing == 0:
        return pts, True

    # Trường hợp 1: Đúng 1 góc bị thiếu/kém -> Ước tính từ 3 góc còn lại
    if n_missing == 1:
        lost_idx = int(np.where(missing)[0][0])
        visible_idx = np.where(~missing)[0]
        pts[lost_idx] = _estimate_one_corner(pts, lost_idx, visible_idx)

        # Kiểm tra hình học góc suy ra có hợp lệ không (tỷ lệ aspect ratio hợp lý)
        aspect = _compute_aspect_ratio_simple(pts)
        if 1.0 <= aspect <= 2.5:
            return pts, True
        else:
            # Tỷ lệ biến dạng quá mức -> không hợp lệ
            return pts, False

    # Trường hợp 2: Bị thiếu >= 2 góc -> Từ chối nắn ảnh (tránh nắn ra ảnh méo vô nghĩa)
    return pts, False


def _estimate_one_corner(
    pts: np.ndarray, lost: int, visible: np.ndarray
) -> np.ndarray:
    """Ước tính vị trí 1 góc bị thiếu dựa trên quy tắc hình bình hành (TL + BR = TR + BL)."""
    # 0: TL, 1: TR, 2: BR, 3: BL
    # TL = TR + BL - BR
    # TR = TL + BR - BL
    # BR = TR + BL - TL
    # BL = TL + BR - TR
    idx_map = {
        0: (1, 3, 2),  # TL = TR + BL - BR
        1: (0, 2, 3),  # TR = TL + BR - BL
        2: (1, 3, 0),  # BR = TR + BL - TL
        3: (0, 2, 1),  # BL = TL + BR - TR
    }
    a, b, c = idx_map[lost]
    return pts[a] + pts[b] - pts[c]


def _compute_aspect_ratio_simple(corners: np.ndarray) -> float:
    """Tính tỷ lệ w/h đơn giản của 4 góc."""
    top_w   = np.linalg.norm(corners[1] - corners[0])
    bot_w   = np.linalg.norm(corners[2] - corners[3])
    left_h  = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])
    avg_w = (top_w + bot_w) / 2.0
    avg_h = (left_h + right_h) / 2.0
    return float(avg_w / avg_h) if avg_h > 0 else 0.0
