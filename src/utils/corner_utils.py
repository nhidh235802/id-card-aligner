"""
corner_utils.py – Các hàm tính toán liên quan đến 4 góc thẻ.
"""

import numpy as np
import cv2


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm theo thứ tự [TL, TR, BR, BL].

    Args:
        pts : (4, 2) array – thứ tự bất kỳ

    Returns:
        ordered : (4, 2) array – [TL, TR, BR, BL]
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)]   # BR: x+y lớn nhất

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR: y-x nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # BL: y-x lớn nhất

    return rect


def compute_angle(corners: np.ndarray) -> float:
    """
    Ước tính góc xoay của thẻ (độ) từ 4 góc.
    Dương = nghiêng phải (theo chiều kim đồng hồ).

    Cách tính: góc của cạnh trên TL→TR so với trục nằm ngang.
    """
    tl, tr = corners[0], corners[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return angle


def compute_aspect_ratio(corners: np.ndarray) -> float:
    """Tính tỷ lệ w/h từ 4 góc thẻ."""
    tl, tr, br, bl = corners
    w = float(np.mean([
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    ]))
    h = float(np.mean([
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    ]))
    return w / h if h > 0 else 1.0


def obb_to_corners(xywhr: np.ndarray) -> np.ndarray:
    """
    Chuyển OBB (cx, cy, w, h, angle_rad) → 4 góc (4, 2).

    Args:
        xywhr : array [cx, cy, w, h, angle_rad]

    Returns:
        corners : (4, 2) float32 – thứ tự [TL, TR, BR, BL] sau khi order
    """
    cx, cy, w, h, angle = xywhr
    box = cv2.boxPoints(((cx, cy), (w, h), float(np.degrees(angle))))
    return box.astype(np.float32)
