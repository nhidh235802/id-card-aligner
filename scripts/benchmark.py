"""
benchmark.py – So sánh 3 phương pháp trên tập test.

Ví dụ:
    python scripts/benchmark.py --data data/splits/test --gt annotations.json
"""

import argparse
import json
import time
import yaml
import cv2
import numpy as np
from pathlib import Path

from src.detector.pose_detector import PoseDetector
from src.detector.obb_detector import OBBDetector
from src.detector.classical_detector import ClassicalDetector
from src.evaluation.metrics import corner_distance_error, angle_error, compute_iou_polygon, benchmark_summary


def run_benchmark(detector, images_dir: str, gt_data: dict) -> dict:
    results = []
    times   = []

    for img_name, gt_corners in gt_data.items():
        img_path = Path(images_dir) / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        t0 = time.perf_counter()
        result = detector.detect(image)
        t1 = time.perf_counter()

        gt = np.array(gt_corners, dtype=np.float32)
        metrics = corner_distance_error(result.corners, gt)
        metrics["angle_err_deg"] = angle_error(result.angle_deg, 0.0)  # gt_angle cần bổ sung
        metrics["iou"] = compute_iou_polygon(result.corners, gt)
        results.append(metrics)
        times.append(t1 - t0)

    summary = benchmark_summary(results)
    summary["mean_latency_ms"] = float(np.mean(times) * 1000)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True)
    parser.add_argument("--gt",      required=True, help="JSON file with ground truth corners")
    parser.add_argument("--methods", nargs="+", default=["classical", "obb", "pose"])
    args = parser.parse_args()

    with open(args.gt) as f:
        gt_data = json.load(f)

    method_map = {
        "pose":      (PoseDetector,      "configs/pose_detector.yaml"),
        "obb":       (OBBDetector,       "configs/obb_detector.yaml"),
        "classical": (ClassicalDetector, "configs/classical_detector.yaml"),
    }

    print(f"\n{'Method':<12} {'MeanErr(px)':<14} {'AngleErr(°)':<14} {'IoU':<8} {'Latency(ms)':<12}")
    print("─" * 60)

    for method in args.methods:
        cls, cfg_path = method_map[method]
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        detector = cls(config).load_model()
        summary  = run_benchmark(detector, args.data, gt_data)

        print(
            f"{method:<12} "
            f"{summary['corner_mean_err_px']:<14.2f} "
            f"{summary['angle_mean_err_deg']:<14.2f} "
            f"{summary['mean_iou']:<8.3f} "
            f"{summary['mean_latency_ms']:<12.1f}"
        )


if __name__ == "__main__":
    main()
