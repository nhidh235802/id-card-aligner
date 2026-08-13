import numpy as np
import cv2


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm góc bất kỳ theo đúng thứ tự chuẩn [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # ── Khối 1: Tính tổng x + y để xác định góc TL (x+y nhỏ nhất) và BR (x+y lớn nhất) ──
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: Top-Left
    rect[2] = pts[np.argmax(s)]   # BR: Bottom-Right

    # ── Khối 2: Tính hiệu y - x để xác định góc TR (y-x nhỏ nhất) và BL (y-x lớn nhất) ──
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR: Top-Right
    rect[3] = pts[np.argmax(diff)]  # BL: Bottom-Left

    return rect


def compute_angle(corners: np.ndarray) -> float:
    """Ước tính góc xoay của thẻ (tính bằng độ) từ góc nghiêng của cạnh trên TL→TR so với phương ngang."""
    tl, tr = corners[0], corners[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return angle


def compute_aspect_ratio(corners: np.ndarray) -> float:
    """Tính toán tỷ lệ giữa chiều rộng và chiều cao (width / height) từ 4 đỉnh góc của thẻ."""
    tl, tr, br, bl = corners
    # Chiều rộng trung bình của cạnh trên và cạnh dưới
    w = float(np.mean([
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    ]))
    # Chiều cao trung bình của cạnh trái và cạnh phải
    h = float(np.mean([
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    ]))
    return w / h if h > 0 else 1.0


def obb_to_corners(xywhr: np.ndarray) -> np.ndarray:
    """Chuyển đổi thông số hộp OBB [cx, cy, w, h, angle_rad] sang mảng tọa độ 4 góc (4, 2)."""
    cx, cy, w, h, angle = xywhr
    # Dùng cv2.boxPoints để lấy 4 đỉnh góc từ thông số hình chữ nhật xoay
    box = cv2.boxPoints(((cx, cy), (w, h), float(np.degrees(angle))))
    return box.astype(np.float32)
