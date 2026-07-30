from .base import BaseDetector
from .pose_detector import PoseDetector
from .obb_detector import OBBDetector
from .classical_detector import ClassicalDetector

__all__ = ["BaseDetector", "PoseDetector", "OBBDetector", "ClassicalDetector"]
