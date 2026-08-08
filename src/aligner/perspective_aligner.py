"""
PerspectiveAligner – nhận 4 góc thẻ [TL, TR, BR, BL] → warp về hình chữ nhật chuẩn ISO ID-1.
"""

import cv2
import numpy as np

CCCD_WIDTH_PX = 856
CCCD_HEIGHT_PX = 540


class PerspectiveAligner:
    """
    Warp thẻ về mặt phẳng chuẩn ISO ID-1 (856 × 540 px) dùng getPerspectiveTransform.

    Args:
        target_width  (int): width ảnh output (mặc định 856)
        target_height (int): height ảnh output (mặc định 540)
    """

    def __init__(
        self,
        target_width: int = CCCD_WIDTH_PX,
        target_height: int = CCCD_HEIGHT_PX,
    ):
        self.target_width = target_width
        self.target_height = target_height

    def align(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Args:
            image   : BGR image (H, W, 3)
            corners : (4, 2) float32 theo thứ tự [TL, TR, BR, BL]

        Returns:
            warped  : BGR image đã align (target_height, target_width, 3)
        """
        src_pts = corners.astype(np.float32)

        dst_pts = np.array([
            [0,                     0                    ],
            [self.target_width - 1, 0                    ],
            [self.target_width - 1, self.target_height - 1],
            [0,                     self.target_height - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            image, M, (self.target_width, self.target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return warped
