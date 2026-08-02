"""
convert_gt_to_yolo.py – Chuyển gt_annotations.json → YOLO label (.txt)

Tạo ra 2 bộ label song song:
  data/yolo_obb/   – dùng để train/eval YOLO-OBB
  data/yolo_pose/  – dùng để train/eval YOLO-Pose

Cấu trúc output:
  data/yolo_obb/
  ├── images/train/   images/val/
  ├── labels/train/   labels/val/
  └── dataset.yaml

  data/yolo_pose/
  ├── images/train/   images/val/
  ├── labels/train/   labels/val/
  └── dataset.yaml

Cách dùng:
  python scripts/convert_gt_to_yolo.py --src data/train_front data/train_back
  python scripts/convert_gt_to_yolo.py --src data/train_front data/train_back --val_ratio 0.2
"""

import json
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
import cv2


# ── Helpers ────────────────────────────────────────────────────────────────────

def corners_to_bbox(corners: list, img_w: int, img_h: int):
    """Tính bbox (cx, cy, w, h) normalize từ 4 góc pixel."""
    pts = np.array(corners, dtype=np.float32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return cx, cy, w, h


def corners_to_obb_label(corners: list, img_w: int, img_h: int) -> str:
    """
    YOLO-OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    Thứ tự corners: [TL, TR, BR, BL] → đi theo CW, hợp lệ với OBB.
    """
    pts = np.array(corners, dtype=np.float32)
    normalized = pts / np.array([img_w, img_h], dtype=np.float32)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
    return f"0 {coords}"


def corners_to_pose_label(corners: list, img_w: int, img_h: int) -> str:
    """
    YOLO-Pose format: class_id cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    Keypoint order: [TL, TR, BR, BL] — PHẢI nhất quán toàn dataset!
    visibility=2: keypoint nhìn thấy rõ (dùng cho ảnh không bị che)
    visibility=1: bị che một phần (category occlusion)
    """
    cx, cy, bw, bh = corners_to_bbox(corners, img_w, img_h)
    pts = np.array(corners, dtype=np.float32)
    normalized = pts / np.array([img_w, img_h], dtype=np.float32)
    kpt_str = "  ".join(f"{x:.6f} {y:.6f} 2" for x, y in normalized)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {kpt_str}"


def get_image_size(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return 900, 900  # fallback canvas size
    return img.shape[1], img.shape[0]  # w, h


# ── Main ───────────────────────────────────────────────────────────────────────

def convert_sources(src_dirs: list, obb_out: str, pose_out: str, val_ratio: float):
    all_items = []  # list of (img_path, corners, is_occluded)

    for src_dir in src_dirs:
        src = Path(src_dir)
        gt_path = src / "gt_annotations.json"
        if not gt_path.exists():
            print(f"  ⚠️  Bỏ qua '{src}' — không có gt_annotations.json")
            continue

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        for rel_path, entry in gt.items():
            img_path = src / rel_path
            if img_path.exists():
                all_items.append((img_path, entry["corners"], entry.get("is_occluded", False)))

    print(f"\nTổng số ảnh thu thập được: {len(all_items)}")

    # Shuffle & split train/val
    random.shuffle(all_items)
    n_val   = int(len(all_items) * val_ratio)
    val_items   = all_items[:n_val]
    train_items = all_items[n_val:]
    print(f"  Train: {len(train_items)} | Val: {len(val_items)}")

    for task, out_dir in [("obb", Path(obb_out)), ("pose", Path(pose_out))]:
        for split, items in [("train", train_items), ("val", val_items)]:
            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, corners, is_occluded in items:
                # Đặt tên file unique để tránh trùng giữa front/back
                src_name = img_path.parent.parent.name  # e.g. "train_front"
                new_stem = f"{src_name}__{img_path.parent.name}__{img_path.stem}"
                dst_img  = img_dir / f"{new_stem}.jpg"
                dst_lbl  = lbl_dir / f"{new_stem}.txt"

                shutil.copy2(img_path, dst_img)

                img_w, img_h = get_image_size(img_path)

                if task == "obb":
                    label_line = corners_to_obb_label(corners, img_w, img_h)
                else:
                    label_line = corners_to_pose_label(corners, img_w, img_h)

                dst_lbl.write_text(label_line + "\n", encoding="utf-8")

        # Tạo dataset.yaml
        kpt_cfg = "\nkpt_shape: [4, 3]  # 4 keypoints, mỗi kpt có (x, y, visibility)" if task == "pose" else ""
        yaml_content = f"""# YOLO-{task.upper()} Dataset — ID Card Aligner
path: {out_dir.resolve().as_posix()}
train: images/train
val:   images/val

nc: 1
names: ['id_card']
{kpt_cfg}
"""
        (out_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
        print(f"  ✅ YOLO-{task.upper()} dataset tạo xong tại: {out_dir}")

    print("\n✅ Hoàn tất convert! Bạn có thể train ngay.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",       nargs="+", required=True,
                        help="Thư mục train (vd: data/train_front data/train_back)")
    parser.add_argument("--obb_out",   default="data/yolo_obb",  help="Output thư mục YOLO-OBB")
    parser.add_argument("--pose_out",  default="data/yolo_pose", help="Output thư mục YOLO-Pose")
    parser.add_argument("--val_ratio", type=float, default=0.2,  help="Tỷ lệ val split (mặc định 0.2)")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    convert_sources(args.src, args.obb_out, args.pose_out, args.val_ratio)
