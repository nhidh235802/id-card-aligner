"""
scripts/measure_report_metrics.py
════════════════════════════════════════════════════════════════════════
Đo các chỉ số cho báo cáo:
  1. Inference speed   — mean ± std (ms/ảnh), FPS
  2. Alignment quality — 2 metric:
       a. Aspect ratio error  : độ lệch tỷ lệ w/h của tứ giác detect được
                                 so với tỷ lệ chuẩn thẻ (856/540 = 1.585)
       b. Rectangularity score: mức độ 4 góc gần 90° (thẻ thật là HCN)
                                 0 = hoàn hảo, cao = méo

Cách dùng:
  python scripts/measure_report_metrics.py
  python scripts/measure_report_metrics.py --warmup 5 --repeat 3
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Tỷ lệ chuẩn của ảnh aligned output (pixel)
TARGET_W, TARGET_H = 856, 540
TARGET_RATIO = TARGET_W / TARGET_H          # 1.5852...

CORNER_NAMES = ["TL", "TR", "BR", "BL"]


# ══════════════════════════════════════════════════════════════════════
#  Geometric alignment metrics từ 4 keypoints
# ══════════════════════════════════════════════════════════════════════

def quad_width_height(pts):
    """
    pts: list of 4 (x,y) theo thứ tự TL TR BR BL.
    Trả (width_top, width_bot, height_left, height_right).
    """
    tl, tr, br, bl = [np.array(p) for p in pts]
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left  = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    return w_top, w_bot, h_left, h_right


def aspect_ratio_error(pts):
    """
    Đo độ lệch tỷ lệ w/h của tứ giác phát hiện so với tỷ lệ chuẩn.
    = |detected_ratio - 1.585| / 1.585  (%)
    """
    w_top, w_bot, h_left, h_right = quad_width_height(pts)
    w = (w_top + w_bot) / 2
    h = (h_left + h_right) / 2
    if h < 1:
        return float("nan")
    ratio = w / h
    return abs(ratio - TARGET_RATIO) / TARGET_RATIO * 100  # percent


def rectangularity_error(pts):
    """
    Đo mức độ 4 góc lệch khỏi 90°.
    Thẻ phải là HCN → 4 góc = 90°. Sai số càng nhỏ càng tốt.
    Trả mean angle deviation (degrees) qua 4 góc.
    """
    tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in pts]
    corners = [tl, tr, br, bl]
    vecs = [
        (tr - tl, bl - tl),   # TL: TR-TL vs BL-TL
        (tl - tr, br - tr),   # TR
        (tr - br, bl - br),   # BR
        (tl - bl, br - bl),   # BL
    ]
    deviations = []
    for v1, v2 in vecs:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        deviations.append(abs(angle - 90.0))
    return np.mean(deviations) if deviations else float("nan")


# ══════════════════════════════════════════════════════════════════════
#  Inference speed
# ══════════════════════════════════════════════════════════════════════

def measure_speed(model: YOLO, images: list, conf: float,
                  warmup: int = 5, repeat: int = 1):
    """
    Đo inference time trên danh sách ảnh.
    warmup: số lần chạy khởi động (không tính)
    repeat: số vòng lặp toàn bộ danh sách (lấy mean)
    Trả (mean_ms, std_ms, fps).
    """
    print(f"  Warmup {warmup} lần...")
    for img_path in images[:warmup]:
        img = cv2.imread(str(img_path))
        if img is not None:
            model.predict(img, conf=conf, verbose=False)

    print(f"  Đo tốc độ ({len(images)} ảnh × {repeat} lần)...")
    times_ms = []
    for _ in range(repeat):
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            t0 = time.perf_counter()
            model.predict(img, conf=conf, verbose=False)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    mean_ms = np.mean(times_ms)
    std_ms  = np.std(times_ms)
    fps     = 1000 / mean_ms if mean_ms > 0 else 0
    return mean_ms, std_ms, fps


# ══════════════════════════════════════════════════════════════════════
#  Combined benchmark
# ══════════════════════════════════════════════════════════════════════

def run(model_path: str, test_img_dir: Path, conf: float,
        warmup: int, repeat: int):

    print(f"\n  Loading model: {model_path}")
    model = YOLO(model_path)
    images = sorted(p for p in test_img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS)
    print(f"  Test images: {len(images)}")

    # ── Inference speed ──────────────────────────────────────────────
    print("\n  [1] Inference Speed")
    mean_ms, std_ms, fps = measure_speed(model, images, conf, warmup, repeat)

    # ── Alignment quality ────────────────────────────────────────────
    print("\n  [2] Alignment Quality (geometric)")
    ar_errors  = []
    rect_errors = []
    detected = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model.predict(img, conf=conf, verbose=False)
        r = results[0]
        if len(r.boxes) == 0 or r.keypoints is None:
            continue
        kpt_xy   = r.keypoints.xy[0].cpu().numpy()    # (4, 2)
        kpt_conf = r.keypoints.conf[0].cpu().numpy()  # (4,)
        if kpt_conf.min() < 0.25:
            continue

        detected += 1
        pts = [(float(kpt_xy[i, 0]), float(kpt_xy[i, 1])) for i in range(4)]
        ar_errors.append(aspect_ratio_error(pts))
        rect_errors.append(rectangularity_error(pts))

    # ── Kết quả ──────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  REPORT METRICS")
    print(f"{'='*58}")

    print(f"\n  [Inference Speed]")
    print(f"    Mean latency  : {mean_ms:.1f} ± {std_ms:.1f} ms/ảnh")
    print(f"    FPS           : {fps:.1f}")
    print(f"    Images tested : {len(images)}")

    print(f"\n  [Alignment Quality]  ({detected}/{len(images)} ảnh detect được)")

    # Aspect ratio error
    ar_valid = [x for x in ar_errors if not np.isnan(x)]
    if ar_valid:
        print(f"    Aspect ratio error  : {np.mean(ar_valid):.2f}% ± {np.std(ar_valid):.2f}%")
        print(f"      (so với tỷ lệ chuẩn 856/540 = {TARGET_RATIO:.3f})")
        print(f"      Min: {min(ar_valid):.2f}%   Max: {max(ar_valid):.2f}%")

    # Rectangularity
    rect_valid = [x for x in rect_errors if not np.isnan(x)]
    if rect_valid:
        print(f"    Rectangularity error: {np.mean(rect_valid):.2f}° ± {np.std(rect_valid):.2f}°")
        print(f"      (lệch so với 90° lý tưởng — càng nhỏ càng tốt)")
        print(f"      Min: {min(rect_valid):.2f}°   Max: {max(rect_valid):.2f}°")

    print(f"\n{'='*58}\n")

    return {
        "mean_ms": mean_ms, "std_ms": std_ms, "fps": fps,
        "detect_rate": detected / len(images) * 100,
        "ar_error_mean": np.mean(ar_valid) if ar_valid else float("nan"),
        "rect_error_mean": np.mean(rect_valid) if rect_valid else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_img",
        default="data_multiclasses/MultipleCard-Detect.yolo26/test/images",
    )
    parser.add_argument(
        "--model",
        default="C:/Users/Admin/ket-qua-train-mixed-v1/weights/best.pt",
    )
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--warmup",  type=int,   default=5)
    parser.add_argument("--repeat",  type=int,   default=3)
    args = parser.parse_args()

    test_img_dir = PROJECT_ROOT / args.test_img
    if not test_img_dir.exists():
        print(f"ERROR: {test_img_dir} không tồn tại")
        sys.exit(1)

    run(args.model, test_img_dir, args.conf, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
