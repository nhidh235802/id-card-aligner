"""
Demo script – chạy nhanh AlignPipeline trên 1 ảnh.

Ví dụ:
    python scripts/demo.py --image assets/samples/cccd_test.jpg --method pose
"""

import argparse
import yaml
import cv2

from src.detector.pose_detector import PoseDetector
from src.detector.obb_detector import OBBDetector
from src.detector.classical_detector import ClassicalDetector
from src.aligner.perspective_aligner import PerspectiveAligner
from src.pipeline.align_pipeline import AlignPipeline
from src.utils.vis_utils import draw_detection


DETECTOR_MAP = {
    "pose":      (PoseDetector,      "configs/pose_detector.yaml"),
    "obb":       (OBBDetector,       "configs/obb_detector.yaml"),
    "classical": (ClassicalDetector, "configs/classical_detector.yaml"),
}


def main():
    parser = argparse.ArgumentParser(description="ID Card Alignment Demo")
    parser.add_argument("--image",  required=True, help="Đường dẫn ảnh input")
    parser.add_argument("--method", default="pose",
                        choices=list(DETECTOR_MAP.keys()))
    parser.add_argument("--show",   action="store_true",
                        help="Hiển thị kết quả bằng cv2.imshow")
    parser.add_argument("--save",   default="outputs/demo/result.jpg",
                        help="Lưu ảnh output")
    args = parser.parse_args()

    # Load detector
    cls, cfg_path = DETECTOR_MAP[args.method]
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    detector = cls(config).load_model()

    # Build pipeline
    aligner  = PerspectiveAligner()
    pipeline = AlignPipeline(detector, aligner)

    # Run
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.image}")

    out = pipeline.run(image)
    print(f"Success : {out.success}")
    print(f"Angle   : {out.detection.angle_deg:.2f}°")
    print(f"Ratio   : {out.detection.aspect_ratio:.4f}")
    print(f"Corners : {out.detection.corners}")

    vis = draw_detection(image, out.detection)

    cv2.imwrite(args.save, out.aligned_image)
    print(f"Đã lưu aligned image → {args.save}")

    if args.show:
        cv2.imshow("Detection", vis)
        cv2.imshow("Aligned",   out.aligned_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
