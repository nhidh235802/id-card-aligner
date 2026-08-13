import cv2
import numpy as np

# Kích thước khung hình tiêu chuẩn ISO ID-1 (pixel)
CCCD_WIDTH_PX = 856
CCCD_HEIGHT_PX = 540


class PerspectiveAligner:
    """Warp thẻ căn cước về mặt phẳng phẳng chuẩn ISO ID-1 (856 × 540 px) bằng ma trận Perspective Transform."""

    def __init__(
        self,
        target_width: int = CCCD_WIDTH_PX,
        target_height: int = CCCD_HEIGHT_PX,
    ):
        self.target_width = target_width
        self.target_height = target_height

    def align(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Biến đổi hình học (warp) ảnh vùng thẻ về hình chữ nhật chuẩn phẳng."""
        # Tọa độ 4 góc nguồn từ detector [TL, TR, BR, BL]
        src_pts = corners.astype(np.float32)

        # Tọa độ 4 góc phẳng đầu ra mong muốn
        dst_pts = np.array([
            [0,                     0                    ],
            [self.target_width - 1, 0                    ],
            [self.target_width - 1, self.target_height - 1],
            [0,                     self.target_height - 1],
        ], dtype=np.float32)

        # ── Khối 1: Tính ma trận biến đổi 3x3 Perspective Transform ──
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # ── Khối 2: Thực hiện Warp Perspective nắn phẳng ảnh thẻ ──
        warped = cv2.warpPerspective(
            image, M, (self.target_width, self.target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return warped
