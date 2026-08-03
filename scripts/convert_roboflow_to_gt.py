"""
convert_roboflow_to_gt.py – Chuyển Roboflow YOLO Pose export → gt_annotations.json

Format input (Roboflow YOLO Pose):
    class_id cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    (tọa độ normalized [0,1])

Format output (gt_annotations.json của chúng ta):
    {
      "ten_anh.jpg": {
        "corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  ← tọa độ pixel
        "category": "real",
        "is_occluded": true/false
      }
    }

Cách dùng:
    python scripts/convert_roboflow_to_gt.py
    python scripts/convert_roboflow_to_gt.py --src data/real_test_roboflow/test --out data/real_test
"""

import cv2
import json
import shutil
import argparse
import numpy as np
from pathlib import Path


def convert(src_dir: str, out_dir: str):
    src      = Path(src_dir)
    img_dir  = src / "images"
    lbl_dir  = src / "labels"
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_gt   = {}
    n_ok     = 0
    n_skip   = 0

    label_files = sorted(lbl_dir.glob("*.txt"))
    print(f"Tìm thấy {len(label_files)} file label...")

    for lbl_path in label_files:
        # Tìm ảnh tương ứng
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = img_dir / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  ⚠️  Không tìm thấy ảnh cho: {lbl_path.name}")
            n_skip += 1
            continue

        # Đọc kích thước ảnh để denormalize
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Không đọc được ảnh: {img_path.name}")
            n_skip += 1
            continue
        img_h, img_w = img.shape[:2]

        # Đọc label
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            print(f"  ⚠️  File label rỗng: {lbl_path.name}")
            n_skip += 1
            continue

        # Lấy dòng đầu tiên (1 thẻ CCCD / ảnh)
        parts = list(map(float, lines[0].split()))

        # Format: class cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
        # Index:    0   1  2  3 4   5  6  7   8  9 10  11 12 13  14 15 16
        if len(parts) < 17:
            print(f"  ⚠️  Label thiếu dữ liệu ({len(parts)} values): {lbl_path.name}")
            n_skip += 1
            continue

        # Trích xuất 4 keypoints và visibility
        kpts = []
        visibilities = []
        for i in range(4):
            x_norm = parts[5 + i * 3]
            y_norm = parts[6 + i * 3]
            v      = int(parts[7 + i * 3])
            kpts.append([x_norm * img_w, y_norm * img_h])
            visibilities.append(v)

        # Kiểm tra: nếu tất cả visibility = 0 → không có nhãn hợp lệ
        if all(v == 0 for v in visibilities):
            print(f"  ⚠️  Tất cả keypoints visibility=0: {lbl_path.name}")
            n_skip += 1
            continue

        # Xác định có bị che khuất không (v=1 = occluded)
        is_occluded = any(v == 1 for v in visibilities)

        # Copy ảnh sang thư mục out với tên đơn giản hơn
        # (bỏ phần suffix ".rf.XXXXXX" mà Roboflow thêm vào)
        clean_name = lbl_path.stem.split("_jpg.rf.")[0] \
                              .split("_jpeg.rf.")[0] \
                              .split("_png.rf.")[0] + ".jpg"
        dst_img = out / clean_name

        # Tránh trùng tên
        counter = 1
        while dst_img.exists():
            stem = clean_name.replace(".jpg", "")
            dst_img = out / f"{stem}_{counter}.jpg"
            counter += 1

        shutil.copy2(img_path, dst_img)

        all_gt[dst_img.name] = {
            "corners":     kpts,
            "category":    "real",
            "is_occluded": is_occluded,
            "visibility":  visibilities,  # Giữ lại để debug
        }
        n_ok += 1
        print(f"  ✅ {dst_img.name}  | occluded={is_occluded} | vis={visibilities}")

    # Lưu GT
    gt_path = out / "gt_annotations.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*55}")
    print(f"  ✅ Thành công: {n_ok} ảnh")
    print(f"  ⚠️  Bỏ qua:    {n_skip} ảnh")
    print(f"  GT lưu tại:  {gt_path}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/real_test_roboflow/test",
                        help="Thư mục Roboflow export (chứa images/ và labels/)")
    parser.add_argument("--out", default="data/real_test",
                        help="Thư mục output (chứa ảnh + gt_annotations.json)")
    args = parser.parse_args()
    convert(args.src, args.out)
