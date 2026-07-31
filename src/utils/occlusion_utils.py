"""
occlusion_utils.py – Xử lý trường hợp thẻ bị che khuất / mất góc.

Chiến lược:
- Nếu 1 góc bị mất (confidence thấp) → ước tính từ 3 góc còn lại
- Nếu 2 góc bị mất cùng cạnh → ước tính từ cạnh đối diện + aspect ratio
- Nếu > 2 góc bị mất → trả về None (không thể recover)
"""

import numpy as np


CCCD_ASPECT_RATIO = 85.6 / 54.0


def handle_missing_corners(
    corners: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Ước tính lại các góc bị mất (confidence < threshold).

    Args:
        corners     : (4, 2) array – [TL, TR, BR, BL]
        confidences : (4,) array   – confidence mỗi góc
        threshold   : float        – ngưỡng xem là "bị mất"

    Returns:
        corners_fixed : (4, 2) array với các góc đã được ước tính
    """
    pts = corners.copy()
    missing = confidences < threshold
    n_missing = int(missing.sum())

    if n_missing == 0:
        return pts
    if n_missing > 2:
        # Không thể recover an toàn, trả về nguyên bản
        return pts

    visible_idx = np.where(~missing)[0]

    if n_missing == 1:
        lost_idx = int(np.where(missing)[0][0])
        pts[lost_idx] = _estimate_one_corner(pts, lost_idx, visible_idx)

    elif n_missing == 2:
        lost_idx = list(np.where(missing)[0])
        pts = _estimate_two_corners(pts, lost_idx, visible_idx)

    return pts


def _estimate_one_corner(
    pts: np.ndarray, lost: int, visible: np.ndarray
) -> np.ndarray:
    """
    Ước tính 1 góc bị mất từ 3 góc còn lại.
    Dùng tính chất parallelogram: TL + BR = TR + BL (midpoint bằng nhau).
    """
    idx_map = {0: (1, 3, 2), 1: (0, 2, 3), 2: (3, 1, 0), 3: (2, 0, 1)}
    a, b, c = idx_map[lost]
    # Góc mất = a + c - b  (parallelogram rule)
    return pts[a] + pts[c] - pts[b]


def _estimate_two_corners(
    pts: np.ndarray, lost: list, visible: list
) -> np.ndarray:
    """
    Ước tính 2 góc bị mất.
    Nếu 2 góc mất cùng cạnh → dùng cạnh đối diện + dịch chuyển theo aspect ratio.
    """
    # Cặp góc cùng cạnh: (0,1)=top, (2,3)=bottom, (0,3)=left, (1,2)=right
    same_edge_pairs = [(0, 1), (2, 3), (0, 3), (1, 2)]
    lost_set = set(lost)

    for pair in same_edge_pairs:
        if lost_set == set(pair):
            # Lấy 2 góc đối diện
            opp = [i for i in range(4) if i not in pair]
            v0, v1 = pts[opp[0]], pts[opp[1]]

            # Vector dịch chuyển từ cạnh đối diện (ước tính chiều cao thẻ)
            h_vec = (v0 + v1) / 2  # midpoint of opposite edge

            if pair == (0, 1):  # top missing, bottom visible
                shift = v0 - v1  # đảo chiều
                pts[0] = v0 - (v1 - v0) * 0  # placeholder
                # Dùng aspect ratio để ước tính
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
    """Trả về vector vuông góc đơn vị với v (xoay 90° CCW)."""
    perp = np.array([-v[1], v[0]], dtype=np.float64)
    norm = np.linalg.norm(perp)
    return perp / norm if norm > 0 else perp
