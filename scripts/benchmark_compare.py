"""
scripts/benchmark_compare.py
════════════════════════════════════════════════════════════════════════
So sánh 2 model YOLO-Pose trên cùng tập test có GT label.

Chỉ số đo:
  1. Detect rate  : % ảnh model phát hiện được (vượt conf threshold)
  2. Pixel error  : Khoảng cách trung bình (pixels) giữa
                    keypoint dự đoán và GT, trên các ảnh detect được
                    (tính theo kích thước ảnh gốc)
  3. Per-corner   : Breakdown lỗi theo từng góc (TL, TR, BR, BL)

GT label format (YOLO-Pose normalized):
  class_id cx cy w h  kx1 ky1 v1  kx2 ky2 v2  kx3 ky3 v3  kx4 ky4 v4

Cách dùng:
  python scripts/benchmark_compare.py

Tùy chỉnh:
  python scripts/benchmark_compare.py \\
      --test_img  data_multiclasses/MultipleCard-Detect.yolo26/test/images \\
      --test_lbl  data_multiclasses/MultipleCard-Detect.yolo26/test/labels \\
      --model_old runs/pose/multiclass_26x/weights/best.pt \\
      --model_new C:/Users/Admin/ket-qua-train-mixed-v1/weights/best.pt \\
      --conf 0.25
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


# ══════════════════════════════════════════════════════════════════════
#  GT Parsing
# ══════════════════════════════════════════════════════════════════════

def parse_gt_label(lbl_path: Path, img_w: int, img_h: int):
    """Đọc GT label → 4 keypoints ở pixel coords."""
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 17:
        return None
    kpts = []
    for i in range(4):
        kx = float(parts[5 + i*3]) * img_w
        ky = float(parts[5 + i*3 + 1]) * img_h
        v  = float(parts[5 + i*3 + 2])
        kpts.append((kx, ky, v))
    return kpts  # list of (x, y, vis)


# ══════════════════════════════════════════════════════════════════════
#  Model Inference
# ══════════════════════════════════════════════════════════════════════

def infer_keypoints(model: YOLO, img: np.ndarray, conf_thresh: float):
    """
    Chạy model, trả về (kpts_px, box_conf) nếu detect được,
    hoặc (None, None) nếu không.
    kpts_px: list of (x, y) — pixel coords
    """
    results = model.predict(img, conf=conf_thresh, verbose=False)
    r = results[0]

    if len(r.boxes) == 0:
        return None, None

    box_conf = float(r.boxes.conf[0])
    if r.keypoints is None or len(r.keypoints.conf) == 0:
        return None, box_conf

    kpt_conf = r.keypoints.conf[0].cpu().numpy()   # (4,)
    kpt_xy   = r.keypoints.xy[0].cpu().numpy()      # (4, 2)

    # Kiểm tra ngưỡng keypoint (dùng min conf giống pipeline)
    if kpt_conf.min() < 0.25:
        return None, box_conf

    kpts_px = [(float(kpt_xy[i, 0]), float(kpt_xy[i, 1])) for i in range(4)]
    return kpts_px, box_conf


# ══════════════════════════════════════════════════════════════════════
#  Benchmark một model
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(model_path: str, test_img_dir: Path,
                  test_lbl_dir: Path, conf_thresh: float, label: str):
    """
    Chạy benchmark, trả về dict kết quả.
    """
    print(f"\n  [{label}] Loading: {model_path}")
    model = YOLO(model_path)

    images = sorted(p for p in test_img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS)

    n_total    = 0
    n_detected = 0
    n_no_gt    = 0
    per_corner_errors = [[] for _ in range(4)]  # TL TR BR BL
    all_mean_errors   = []
    failed_images     = []

    for img_path in images:
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            n_no_gt += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        n_total += 1

        gt_kpts = parse_gt_label(lbl_path, w, h)
        if gt_kpts is None:
            n_no_gt += 1
            continue

        pred_kpts, box_conf = infer_keypoints(model, img, conf_thresh)

        if pred_kpts is None:
            failed_images.append(img_path.name)
            continue

        n_detected += 1

        # Pixel error mỗi góc
        errors = []
        for i in range(4):
            gx, gy, gv = gt_kpts[i]
            px, py     = pred_kpts[i]
            err = np.sqrt((px - gx)**2 + (py - gy)**2)
            per_corner_errors[i].append(err)
            errors.append(err)
        all_mean_errors.append(np.mean(errors))

    detect_rate = n_detected / n_total * 100 if n_total > 0 else 0
    mean_err    = np.mean(all_mean_errors)    if all_mean_errors else float("nan")
    corner_means = [np.mean(e) if e else float("nan")
                    for e in per_corner_errors]

    return {
        "label":        label,
        "n_total":      n_total,
        "n_detected":   n_detected,
        "detect_rate":  detect_rate,
        "mean_err_px":  mean_err,
        "corner_errs":  corner_means,
        "failed":       failed_images,
    }


# ══════════════════════════════════════════════════════════════════════
#  In báo cáo
# ══════════════════════════════════════════════════════════════════════

def print_report(results: list):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPARISON — YOLO-Pose Keypoint Detection")
    print(f"{'='*60}")

    # Header
    labels = [r["label"] for r in results]
    print(f"\n  {'Chỉ số':<28}", end="")
    for r in results:
        print(f"  {r['label'][:16]:>16}", end="")
    print()
    print(f"  {'─'*28}", end="")
    for _ in results:
        print(f"  {'─'*16}", end="")
    print()

    # Detect rate
    print(f"  {'Detect rate':<28}", end="")
    for r in results:
        v = f"{r['n_detected']}/{r['n_total']}  ({r['detect_rate']:.1f}%)"
        print(f"  {v:>16}", end="")
    print()

    # Mean pixel error (overall)
    print(f"  {'Mean pixel error (all kpt)':<28}", end="")
    for r in results:
        v = f"{r['mean_err_px']:.2f} px" if not np.isnan(r['mean_err_px']) else "N/A"
        print(f"  {v:>16}", end="")
    print()

    # Per-corner error
    for ci, cname in enumerate(CORNER_NAMES):
        print(f"  {'  Error ' + cname:<28}", end="")
        for r in results:
            v = f"{r['corner_errs'][ci]:.2f} px" if not np.isnan(r['corner_errs'][ci]) else "N/A"
            print(f"  {v:>16}", end="")
        print()

    # Failed images
    print()
    for r in results:
        if r["failed"]:
            print(f"  [{r['label']}] SKIP ({len(r['failed'])} ảnh):")
            for name in r["failed"]:
                print(f"    - {name[:60]}")

    print(f"\n{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="So sanh 2 model YOLO-Pose tren tap test co GT"
    )
    parser.add_argument(
        "--test_img",
        default="data_multiclasses/MultipleCard-Detect.yolo26/test/images",
        help="Thu muc anh test"
    )
    parser.add_argument(
        "--test_lbl",
        default="data_multiclasses/MultipleCard-Detect.yolo26/test/labels",
        help="Thu muc label GT tuong ung"
    )
    parser.add_argument(
        "--model_old",
        default="runs/pose/multiclass_26x/weights/best.pt",
        help="Model lan train 1 (synthetic only)"
    )
    parser.add_argument(
        "--model_new",
        default="C:/Users/Admin/ket-qua-train-mixed-v1/weights/best.pt",
        help="Model lan train 2 (mixed)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold cho box detection"
    )
    args = parser.parse_args()

    test_img_dir = PROJECT_ROOT / args.test_img
    test_lbl_dir = PROJECT_ROOT / args.test_lbl

    if not test_img_dir.exists():
        print(f"ERROR: Khong tim thay {test_img_dir}")
        sys.exit(1)

    results = []
    for model_path, label in [
        (args.model_old, "Synthetic only"),
        (args.model_new, "Mixed (real aug)"),
    ]:
        if not Path(model_path).exists():
            print(f"  WARN: Khong tim thay model {model_path}, bo qua.")
            continue
        r = run_benchmark(model_path, test_img_dir,
                          test_lbl_dir, args.conf, label)
        results.append(r)

    if results:
        print_report(results)


if __name__ == "__main__":
    main()
