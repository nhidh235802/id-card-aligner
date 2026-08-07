"""
PerspectiveAligner – nhận 4 góc thẻ → warp về hình chữ nhật chuẩn.

Xử lý:
- Perspective transform (homography)
- Aspect ratio correction (CCCD chuẩn = 85.6mm × 54mm ≈ 1.5852)
- Auto-rotate: đảm bảo output luôn là landscape (width > height)
"""

import cv2
import numpy as np

# Tỷ lệ chuẩn thẻ CCCD Việt Nam (ISO/IEC 7810 ID-1)
CCCD_ASPECT_RATIO = 85.6 / 54.0   # ≈ 1.5852
CCCD_WIDTH_PX = 856                # output width mặc định (×10 mm)
CCCD_HEIGHT_PX = 540


class PerspectiveAligner:
    """
    Warp thẻ về mặt phẳng chuẩn dùng getPerspectiveTransform.

    Args:
        target_width  (int)  : width ảnh output (mặc định 856)
        target_height (int)  : height ảnh output (mặc định 540)
        fix_aspect    (bool) : tự động sửa tỷ lệ nếu thẻ bị méo
        auto_rotate   (bool) : tự động xoay về landscape nếu output là portrait
    """

    def __init__(
        self,
        target_width: int = CCCD_WIDTH_PX,
        target_height: int = CCCD_HEIGHT_PX,
        fix_aspect: bool = True,
        auto_rotate: bool = True,
        padding: int = 0,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.fix_aspect = fix_aspect
        self.auto_rotate = auto_rotate
        self.padding = padding

    def align(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Args:
            image   : BGR image (H, W, 3)
            corners : (4, 2) float32 theo thứ tự [TL, TR, BR, BL]

        Returns:
            warped  : BGR image đã align (target_height, target_width, 3)
        """
        src_pts = corners.astype(np.float32)

        w, h = self._compute_output_size(src_pts)

        dst_pts = np.array([
            [0,     0    ],
            [w - 1, 0    ],
            [w - 1, h - 1],
            [0,     h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        # ── Bước 1: Auto-rotate về landscape ──────────────────────────────────
        # CCCD là thẻ ngang (landscape). Nếu warp ra ảnh dọc (portrait)
        # nghĩa là thẻ trong ảnh gốc đang bị chụp xoay 90° → xoay lại.
        if self.auto_rotate and warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        # ── Bước 2: Resize về kích thước chuẩn ISO ────────────────────────────
        if self.fix_aspect:
            warped = cv2.resize(warped, (self.target_width, self.target_height),
                                interpolation=cv2.INTER_AREA)
        return warped

    def _compute_output_size(self, corners: np.ndarray):
        """
        Tính w, h output từ khoảng cách thực tế giữa các góc.
        Luôn đảm bảo w >= h để warp ban đầu không bị quá nhỏ.
        """
        tl, tr, br, bl = corners

        width_top    = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        w = int(max(width_top, width_bottom))

        height_left  = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        h = int(max(height_left, height_right))

        # Đảm bảo dimension lớn hơn luôn là w (cạnh dài nhất = chiều ngang)
        # để warp ban đầu không bị crop nội dung bất kỳ phương hướng nào
        if w < h:
            w, h = h, w

        # Aspect ratio correction: nếu tỷ lệ lệch > 5% → dùng CCCD chuẩn
        if self.fix_aspect:
            detected_ratio = w / h if h > 0 else CCCD_ASPECT_RATIO
            if abs(detected_ratio - CCCD_ASPECT_RATIO) / CCCD_ASPECT_RATIO > 0.05:
                h = int(w / CCCD_ASPECT_RATIO)

        return w, h
