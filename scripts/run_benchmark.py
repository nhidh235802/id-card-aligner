"""
run_benchmark.py – Lấy số liệu so sánh 3 phương pháp cho báo cáo.

Output khớp chính xác với bảng báo cáo:
  - Corner Distance Error (px, trung bình)
  - % ảnh detect thành công
  - Thời gian xử lý / ảnh (ms)
  - Breakdown theo từng category

Cách dùng:
  # Baseline (chưa fine-tune) với Classical CV:
  python scripts/run_benchmark.py --testset data/synthetic_testset_front --methods classical

  # So sánh đầy đủ sau khi fine-tune:
  python scripts/run_benchmark.py ^
    --testset data/synthetic_testset_front ^
    --methods classical obb pose ^
    --obb_weights runs/train/obb_finetune/weights/best.pt ^
    --pose_weights runs/train/pose_finetune/weights/best.pt

  # Test trên real_test set:
  python scripts/run_benchmark.py --testset data/real_test --methods classical obb pose ...
"""

import argparse
import json
import sys
import time
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Tự động thêm thư mục root vào PYTHONPATH để import module 'src' không bị lỗi
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Metrics ────────────────────────────────────────────────────────────────────

def corner_distance_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Euclidean distance giữa 4 góc predict và GT (pixel)."""
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def polygon_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_i = pred.astype(np.float32).reshape(-1, 1, 2)
    gt_i   = gt.astype(np.float32).reshape(-1, 1, 2)
    inter, _  = cv2.intersectConvexConvex(pred_i, gt_i)
    pred_area = cv2.contourArea(pred_i)
    gt_area   = cv2.contourArea(gt_i)
    union = pred_area + gt_area - inter
    return float(inter / union) if union > 0 else 0.0


# ── Detectors ──────────────────────────────────────────────────────────────────

def load_classical(config_path="configs/classical_detector.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.detector.classical_detector import ClassicalDetector
    return ClassicalDetector(cfg).load_model()


def load_obb(weights: str, config_path="configs/obb_detector.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["weights"] = weights
    from src.detector.obb_detector import OBBDetector
    return OBBDetector(cfg).load_model()


def load_pose(weights: str, config_path="configs/pose_detector.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["weights"] = weights
    from src.detector.pose_detector import PoseDetector
    return PoseDetector(cfg).load_model()


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_on_testset(detector, testset_dir: Path, gt_data: dict, conf_threshold: float = 0.3):
    """
    Chạy detector trên toàn bộ testset.
    Returns: dict keyed by category → list of per-image result dicts.
    """
    category_results = defaultdict(list)

    for rel_key, entry in gt_data.items():
        img_path = testset_dir / rel_key
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        gt_corners = np.array(entry["corners"], dtype=np.float32)
        category   = entry.get("category", "unknown")

        t0 = time.perf_counter()
        result = detector.detect(image)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        success = result.confidence >= conf_threshold

        row = {
            "success":    success,
            "latency_ms": latency_ms,
        }

        if success:
            row["corner_err_px"] = corner_distance_error(result.corners, gt_corners)
            row["iou"]           = polygon_iou(result.corners, gt_corners)
        else:
            row["corner_err_px"] = None
            row["iou"]           = None

        category_results[category].append(row)

    return category_results


def summarize(category_results: dict) -> dict:
    """Tổng hợp metrics theo từng category và overall."""
    summary = {}
    all_rows = []

    for cat, rows in sorted(category_results.items()):
        all_rows.extend(rows)
        n_total   = len(rows)
        n_success = sum(1 for r in rows if r["success"])
        errs      = [r["corner_err_px"] for r in rows if r["corner_err_px"] is not None]
        latencies = [r["latency_ms"] for r in rows]

        summary[cat] = {
            "n":               n_total,
            "detect_rate_pct": round(100 * n_success / n_total, 1) if n_total else 0,
            "corner_err_px":   round(float(np.mean(errs)), 2) if errs else None,
            "latency_ms":      round(float(np.mean(latencies)), 1),
        }

    # Overall
    n_total   = len(all_rows)
    n_success = sum(1 for r in all_rows if r["success"])
    errs      = [r["corner_err_px"] for r in all_rows if r["corner_err_px"] is not None]
    latencies = [r["latency_ms"] for r in all_rows]

    summary["__overall__"] = {
        "n":               n_total,
        "detect_rate_pct": round(100 * n_success / n_total, 1) if n_total else 0,
        "corner_err_px":   round(float(np.mean(errs)), 2) if errs else None,
        "latency_ms":      round(float(np.mean(latencies)), 1),
    }

    return summary


def print_table(method_summaries: dict):
    """In bảng so sánh khớp với format báo cáo."""
    methods = list(method_summaries.keys())

    # ── Bảng tổng quan (6.2) ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  Bảng 6.2 – So sánh định lượng tổng thể")
    print(f"{'─'*72}")
    header = f"  {'Phương pháp':<22} {'Corner Err (px)':<18} {'Detect % ':<14} {'Latency (ms)'}"
    print(header)
    print(f"  {'-'*68}")

    for method, summaries in method_summaries.items():
        ov = summaries["__overall__"]
        err_str = f"{ov['corner_err_px']:.2f}" if ov['corner_err_px'] is not None else "N/A"
        print(f"  {method:<22} {err_str:<18} {ov['detect_rate_pct']:<14} {ov['latency_ms']}")

    # ── Bảng chi tiết theo category (6.3) ────────────────────────────────────
    cats = [c for c in list(list(method_summaries.values())[0].keys()) if c != "__overall__"]

    print(f"\n{'─'*90}")
    print(f"  Bảng 6.3 – Kết quả theo từng category (Corner Err px | Detect %)")
    print(f"{'─'*90}")
    meth_header = "".join(f"  {m:<28}" for m in methods)
    print(f"  {'Category':<22}{meth_header}")
    print(f"  {'-'*86}")

    for cat in cats:
        cat_label = cat.replace("_", " ").replace("1 ", "").replace("2 ", "").replace(
            "3 ", "").replace("4 ", "").replace("5 ", "").replace("6 ", "")
        row = f"  {cat_label:<22}"
        for method in methods:
            s = method_summaries[method].get(cat, {})
            if not s:
                row += f"  {'—':<28}"
                continue
            err = f"{s['corner_err_px']:.1f}px" if s['corner_err_px'] is not None else "FAIL"
            row += f"  {err} | {s['detect_rate_pct']}%{'':<14}"
        print(row)

    print(f"{'─'*90}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 3 phương pháp detect thẻ CCCD")
    parser.add_argument("--testset",      required=True,
                        help="Thư mục testset (phải có gt_annotations.json)")
    parser.add_argument("--methods",      nargs="+",
                        choices=["classical", "obb", "pose"], default=["classical"],
                        help="Phương pháp cần chạy")
    parser.add_argument("--obb_weights",  default="yolo11n-obb.pt",
                        help="Weights YOLO-OBB (pretrained hoặc fine-tuned)")
    parser.add_argument("--pose_weights", default="yolo11n-pose.pt",
                        help="Weights YOLO-Pose (pretrained hoặc fine-tuned)")
    parser.add_argument("--conf",         type=float, default=0.3,
                        help="Ngưỡng confidence để coi là detect thành công")
    parser.add_argument("--save_json",    default=None,
                        help="Lưu kết quả ra file JSON (tuỳ chọn)")
    args = parser.parse_args()

    # Load GT
    testset_dir = Path(args.testset)
    gt_path     = testset_dir / "gt_annotations.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {gt_path}")
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"Testset: {testset_dir} ({len(gt_data)} ảnh)")

    # Load detectors & run
    all_summaries = {}

    if "classical" in args.methods:
        print("\n▶ Chạy Classical CV...")
        det = load_classical()
        cat_results = run_on_testset(det, testset_dir, gt_data, args.conf)
        all_summaries["OpenCV Contour"] = summarize(cat_results)

    if "obb" in args.methods:
        print(f"\n▶ Chạy YOLO-OBB ({args.obb_weights})...")
        det = load_obb(args.obb_weights)
        cat_results = run_on_testset(det, testset_dir, gt_data, args.conf)
        label = "YOLO-OBB (pretrain)" if "yolo11n" in args.obb_weights else "YOLO-OBB (finetune)"
        all_summaries[label] = summarize(cat_results)

    if "pose" in args.methods:
        print(f"\n▶ Chạy YOLO-Pose ({args.pose_weights})...")
        det = load_pose(args.pose_weights)
        cat_results = run_on_testset(det, testset_dir, gt_data, args.conf)
        label = "YOLO-Pose (pretrain)" if "yolo11n" in args.pose_weights else "YOLO-Pose (finetune)"
        all_summaries[label] = summarize(cat_results)

    # Print tables
    print_table(all_summaries)

    # Lưu JSON nếu cần
    if args.save_json:
        import json as _json
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            _json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"Kết quả đã lưu → {args.save_json}")
