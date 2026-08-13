import numpy as np

# Tỷ lệ khung hình thẻ CCCD chuẩn (Width / Height)
CCCD_ASPECT_RATIO = 85.6 / 54.0


def handle_missing_corners(
    corners: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Ước tính lại các vị trí điểm góc bị mất hoặc bị ngón tay che khuất (confidence < threshold)."""
    pts = corners.copy()
    missing = confidences < threshold
    n_missing = int(missing.sum())

    # Nếu không có góc nào bị che -> giữ nguyên
    if n_missing == 0:
        return pts
    # Nếu bị che nhiều hơn 2 góc -> không thể khôi phục an toàn, trả về nguyên bản
    if n_missing > 2:
        return pts

    visible_idx = np.where(~missing)[0]

    # Trường hợp 1: Bị che 1 góc -> tính toán lại từ 3 góc còn lại
    if n_missing == 1:
        lost_idx = int(np.where(missing)[0][0])
        pts[lost_idx] = _estimate_one_corner(pts, lost_idx, visible_idx)

    # Trường hợp 2: Bị che 2 góc -> ước tính từ cạnh đối diện và tỷ lệ aspect ratio
    elif n_missing == 2:
        lost_idx = list(np.where(missing)[0])
        pts = _estimate_two_corners(pts, lost_idx, visible_idx)

    return pts


def _estimate_one_corner(
    pts: np.ndarray, lost: int, visible: np.ndarray
) -> np.ndarray:
    """Ước tính vị trí 1 góc bị mất dựa vào tính chất hình bình hành (TL + BR = TR + BL)."""
    idx_map = {0: (1, 3, 2), 1: (0, 2, 3), 2: (3, 1, 0), 3: (2, 0, 1)}
    a, b, c = idx_map[lost]
    # Công thức quy tắc hình bình hành: Góc mất = a + c - b
    return pts[a] + pts[c] - pts[b]


def _estimate_two_corners(
    pts: np.ndarray, lost: list, visible: list
) -> np.ndarray:
    """Ước tính 2 góc bị mất trên cùng 1 cạnh bằng cách dịch chuyển vector cạnh đối diện theo tỷ lệ CCCD_ASPECT_RATIO."""
    same_edge_pairs = [(0, 1), (2, 3), (0, 3), (1, 2)]
    lost_set = set(lost)

    for pair in same_edge_pairs:
        if lost_set == set(pair):
            opp = [i for i in range(4) if i not in pair]
            v0, v1 = pts[opp[0]], pts[opp[1]]

            h_vec = (v0 + v1) / 2

            # Khối tính độ dài cạnh đối diện và suy ra khoảng cách chiều cao cần dịch chuyển
            if pair == (0, 1):
                shift = v0 - v1
                pts[0] = v0 - (v1 - v0) * 0
                edge_len = np.linalg.norm(v1 - v0)
                h_est = edge_len / CCCD_ASPECT_RATIO
                normal = _perpendicular_unit(v1 - v0)
                pts[pair[0]] = v0 - normal * h_est
                pts[pair[1]] = v1 - normal * h_est
            else:
                edge_len = np.linalg.norm(v1 - v0)
                h_est = edge_len / CCCD_ASPECT_RATIO
                normal = _perpendicular_unit(v1 - v0)
                pts[pair[0]] = v0 + normal * h_est
                pts[pair[1]] = v1 + normal * h_est
            break

    return pts


def _perpendicular_unit(v: np.ndarray) -> np.ndarray:
    """Tính toán và trả về vector đơn vị vuông góc với vector v (xoay 90° ngược chiều kim đồng hồ)."""
    perp = np.array([-v[1], v[0]], dtype=np.float64)
    norm = np.linalg.norm(perp)
    return perp / norm if norm > 0 else perp
