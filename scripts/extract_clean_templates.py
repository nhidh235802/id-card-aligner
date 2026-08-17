"""
scripts/extract_clean_templates.py
─────────────────────────────────────────────────────────────────────────────
Đọc ảnh + nhãn YOLO-Pose đã gán (4 keypoints góc), warp perspective để tạo
ảnh thẻ phẳng (clean template) dùng làm đầu vào cho generate_synthetic_testset.

Mỗi ảnh đầu vào → 1 file ảnh phẳng đã căn chỉnh, nhóm theo class.

Cấu trúc output:
    assets/templates/<class_name>/
        ├── img001_clean.jpg
        ├── img002_clean.jpg
        └── ...

Cách dùng:
    # Xử lý toàn bộ split train
    python scripts/extract_clean_templates.py \\
        --images data_multiclasses/MultipleCard-Detect.yolo26/train/images \\
        --labels data_multiclasses/MultipleCard-Detect.yolo26/train/labels \\
        --out_dir assets/templates

    # Xem preview (vẽ overlay 4 góc, không save aligned)
    python scripts/extract_clean_templates.py ... --preview

    # Chỉ xử lý class cụ thể (theo mentor class_id)
    python scripts/extract_clean_templates.py ... --only_class 12 13
"""

import sys
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligner.perspective_aligner import PerspectiveAligner
from src.detector.base import CLASS_NAMES
from src.utils.vis_utils import draw_detection_result
from src.detector.base import DetectionResult

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kích thước warp mặc định: ISO ID-1 (thẻ nhựa chuẩn)
# Nếu Singapore FIN card có tỷ lệ khác thì điều chỉnh ở đây
# Singapore NRIC: 85.6 × 54mm → cùng tỷ lệ ID-1
DEFAULT_WIDTH  = 856
DEFAULT_HEIGHT = 540


# ── Đọc nhãn YOLO-Pose một dòng ─────────────────────────────────────────────

def parse_yolo_pose_label(line: str, img_w: int, img_h: int):
    """Parse 1 dòng nhãn YOLO-Pose keypoint.

    Format: class_id cx cy bw bh  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    Trả về: (class_id: int, corners_pixel: np.ndarray shape (4,2))
    """
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5 + 4 * 3:
        return None, None

    class_id = int(parts[0])

    # Trích 4 keypoints (bỏ qua visibility)
    kpts = []
    for i in range(4):
        base = 5 + i * 3
        x_norm = parts[base]
        y_norm = parts[base + 1]
        # v    = parts[base + 2]  # visibility — không dùng ở đây
        kpts.append([x_norm * img_w, y_norm * img_h])

    corners = np.array(kpts, dtype=np.float32)  # (4, 2) [TL, TR, BR, BL]
    return class_id, corners


# ── Auto-detect kích thước output phù hợp ───────────────────────────────────

def estimate_output_size(corners: np.ndarray):
    """Ước tính chiều rộng/cao thực của thẻ từ 4 góc, giữ nguyên tỷ lệ."""
    # Chiều rộng = trung bình cạnh trên + cạnh dưới
    w_top  = np.linalg.norm(corners[1] - corners[0])
    w_bot  = np.linalg.norm(corners[2] - corners[3])
    h_left = np.linalg.norm(corners[3] - corners[0])
    h_right= np.linalg.norm(corners[2] - corners[1])
    avg_w  = (w_top + w_bot) / 2
    avg_h  = (h_left + h_right) / 2
    return avg_w, avg_h


# ── Warp một ảnh ─────────────────────────────────────────────────────────────

def warp_card(image: np.ndarray, corners: np.ndarray,
              out_width: int, out_height: int) -> np.ndarray:
    """Warp perspective từ 4 góc keypoint → ảnh phẳng."""
    src_pts = corners.astype(np.float32)
    dst_pts = np.array([
        [0,           0           ],
        [out_width-1, 0           ],
        [out_width-1, out_height-1],
        [0,           out_height-1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image, M, (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


# ── Main processing ──────────────────────────────────────────────────────────

def process_dataset(images_dir: Path, labels_dir: Path,
                    out_dir: Path, only_classes: set,
                    out_width: int, out_height: int,
                    preview: bool, dry_run: bool) -> dict:
    """Duyệt qua tất cả ảnh, warp và lưu clean template.

    Returns: dict thống kê {class_name: count}
    """
    img_files = sorted(p for p in images_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTS)

    if not img_files:
        print(f"  ⚠️  Không tìm thấy ảnh nào trong {images_dir}")
        return {}

    stats = {}
    skipped = 0

    for img_path in img_files:
        # Tìm file label tương ứng
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"  ⚠️  Không có label: {img_path.name}")
            skipped += 1
            continue

        # Đọc ảnh
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Không đọc được: {img_path.name}")
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]

        # Đọc tất cả dòng label (thường chỉ có 1 dòng/ảnh)
        lines = [l for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            skipped += 1
            continue

        for line in lines:
            class_id, corners = parse_yolo_pose_label(line, img_w, img_h)
            if class_id is None:
                continue

            # Lọc theo class nếu có --only_class
            if only_classes and class_id not in only_classes:
                continue

            # Tên class
            cname = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class_{class_id}"

            # Preview: vẽ overlay 4 góc lên ảnh gốc
            if preview:
                result = DetectionResult(
                    corners=corners,
                    confidence=1.0,
                    class_id=class_id,
                    bbox_xyxy=None,
                )
                vis = draw_detection_result(image, result, show_corner_labels=True)
                cv2.imshow(f"[{cname}] {img_path.name} — press any key", vis)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == 27:  # ESC
                    return stats

            # Warp → clean template
            # Tự điều chỉnh output size theo tỷ lệ thực của thẻ trong ảnh
            est_w, est_h = estimate_output_size(corners)
            if est_w > est_h:
                final_w, final_h = out_width, out_height
            else:
                # Thẻ chụp dọc (portrait) → swap
                final_w, final_h = out_height, out_width

            warped = warp_card(image, corners, final_w, final_h)

            if not dry_run:
                # Lưu theo class
                class_dir = out_dir / cname
                class_dir.mkdir(parents=True, exist_ok=True)
                out_path = class_dir / f"{img_path.stem}_clean.jpg"
                cv2.imwrite(str(out_path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"  ✔  [{cname}] {img_path.name} → {out_path.name}")

            stats[cname] = stats.get(cname, 0) + 1

    if skipped:
        print(f"\n  (Bỏ qua {skipped} ảnh không có label)")
    return stats


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Warp ảnh thẻ về phẳng dùng keypoint label để tạo clean template"
    )
    parser.add_argument(
        "--images", required=True,
        help="Thư mục chứa ảnh đầu vào"
    )
    parser.add_argument(
        "--labels", default=None,
        help="Thư mục chứa file .txt label (mặc định: cùng cấp với images, tên 'labels')"
    )
    parser.add_argument(
        "--out_dir", default="assets/templates",
        help="Thư mục lưu clean templates (mặc định: assets/templates)"
    )
    parser.add_argument(
        "--width",  type=int, default=DEFAULT_WIDTH,
        help=f"Chiều rộng output pixel (mặc định: {DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=f"Chiều cao output pixel (mặc định: {DEFAULT_HEIGHT})"
    )
    parser.add_argument(
        "--only_class", nargs="+", type=int, default=None,
        help="Chỉ xử lý các class_id này (vd: --only_class 12 13)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Hiển thị overlay 4 góc trước khi warp (nhấn ESC để thoát)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Thống kê mà không ghi file"
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"❌ Không tìm thấy: {images_dir}")
        sys.exit(1)

    # Auto-detect labels dir
    if args.labels:
        labels_dir = Path(args.labels)
    else:
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        if not labels_dir.exists():
            labels_dir = images_dir.parent / "labels"

    if not labels_dir.exists():
        print(f"❌ Không tìm thấy labels dir: {labels_dir}")
        print(f"   Hãy chỉ định rõ với --labels <path>")
        sys.exit(1)

    out_dir     = PROJECT_ROOT / args.out_dir
    only_cls    = set(args.only_class) if args.only_class else set()

    print(f"\n{'='*60}")
    print(f"  Extract Clean Templates from Keypoint Labels")
    if args.dry_run:
        print(f"  [DRY RUN] — không ghi file")
    print(f"  Images : {images_dir}")
    print(f"  Labels : {labels_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Size   : {args.width} x {args.height} px")
    if only_cls:
        cnames = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in only_cls]
        print(f"  Filter : class {only_cls} ({', '.join(cnames)})")
    print(f"{'='*60}\n")

    stats = process_dataset(
        images_dir   = images_dir,
        labels_dir   = labels_dir,
        out_dir      = out_dir,
        only_classes = only_cls,
        out_width    = args.width,
        out_height   = args.height,
        preview      = args.preview,
        dry_run      = args.dry_run,
    )

    # Báo cáo kết quả
    print(f"\n{'='*60}")
    print(f"  Kết quả:")
    total = sum(stats.values())
    for cname, count in sorted(stats.items()):
        print(f"    {cname:<16} : {count} ảnh")
    print(f"  TỔNG               : {total} clean templates")
    if not args.dry_run and total > 0:
        print(f"\n  ✅ Templates lưu tại: {out_dir}")
        print(f"\n  Bước tiếp theo — cập nhật configs/multiclass_sources.yaml:")
        for cname in sorted(stats.keys()):
            print(f"    template: \"assets/templates/{cname}/<chọn_1_ảnh_tốt_nhất>.jpg\"")
        print(f"\n  Sau đó chạy:")
        print(f"    python scripts/build_multiclass_dataset.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
