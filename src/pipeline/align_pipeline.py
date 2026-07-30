"""
AlignPipeline – kết nối Detector + Aligner thành pipeline hoàn chỉnh.

Input : ảnh thô (BGR)
Output: ảnh thẻ đã align (BGR) + DetectionResult

Thiết kế để dễ tích hợp Multi-task model:
  - Detector là interface, swap model không cần sửa pipeline
  - DetectionResult.extra_heads chứa output của head phụ (nếu có)
"""

import numpy as np
from dataclasses import dataclass

from src.detector.base import BaseDetector, DetectionResult
from src.aligner.perspective_aligner import PerspectiveAligner


@dataclass
class PipelineOutput:
    aligned_image: np.ndarray       # ảnh đã warp (H, W, 3)
    detection: DetectionResult      # kết quả detect đầy đủ
    success: bool                   # False nếu không detect được thẻ


class AlignPipeline:
    """
    End-to-end pipeline: raw image → aligned card image.

    Ví dụ sử dụng:
        detector = PoseDetector(config).load_model()
        aligner  = PerspectiveAligner()
        pipeline = AlignPipeline(detector, aligner)
        out = pipeline.run(image)
        cv2.imwrite("aligned.jpg", out.aligned_image)
    """

    def __init__(
        self,
        detector: BaseDetector,
        aligner: PerspectiveAligner,
        min_confidence: float = 0.4,
    ):
        self.detector = detector
        self.aligner = aligner
        self.min_confidence = min_confidence

    def run(self, image: np.ndarray) -> PipelineOutput:
        """
        Args:
            image : BGR numpy array

        Returns:
            PipelineOutput
        """
        detection = self.detector.detect(image)

        if detection.confidence < self.min_confidence:
            return PipelineOutput(
                aligned_image=image,
                detection=detection,
                success=False,
            )

        aligned = self.aligner.align(image, detection.corners)

        return PipelineOutput(
            aligned_image=aligned,
            detection=detection,
            success=True,
        )
