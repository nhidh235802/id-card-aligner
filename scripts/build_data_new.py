"""
build_data_new.py – Tự động tạo bộ dữ liệu mới tại thư mục `data_new/`

Cấu trúc tạo ra:
  data_new/
  ├── train_front/             (360 ảnh = 60 ảnh × 6 categories)
  ├── val_front/               (120 ảnh = 20 ảnh × 6 categories)
  ├── synthetic_testset_front/ (120 ảnh = 20 ảnh × 6 categories)
  ├── train_back/              (360 ảnh = 60 ảnh × 6 categories)
  ├── val_back/                (120 ảnh = 20 ảnh × 6 categories)
  ├── synthetic_testset_back/  (120 ảnh = 20 ảnh × 6 categories)
  ├── real_test/               (30 ảnh thực tế copy từ data/real_test)
  ├── yolo_obb/                (Dataset YOLO-OBB train: 720, val: 240)
  └── yolo_pose/               (Dataset YOLO-Pose train: 720, val: 240)

Cách dùng:
  python scripts/build_data_new.py
"""

import os
import sys
import shutil
import json
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_testset import generate_comprehensive_dataset
from scripts.convert_gt_to_yolo import corners_to_obb_label, corners_to_pose_label, corners_to_bbox


def copy_real_test(src_real: Path, dst_real: Path):
    """Copy 30 ảnh real test và gt_annotations.json sang data_new/real_test/"""
    dst_real.mkdir(parents=True, exist_ok=True)

    if not src_real.exists():
        print(f"⚠️  Thư mục nguồn '{src_real}' không tồn tại!")
        return

    count = 0
    for item in src_real.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_real / item.name)
            if item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                count += 1

    print(f"✅ Đã copy {count} ảnh real test + gt_annotations.json sang '{dst_real}'")


def convert_dataset_to_yolo(train_dirs: list, val_dirs: list, obb_dir: Path, pose_dir: Path):
    """Convert train_dirs và val_dirs thành YOLO-OBB và YOLO-Pose datasets."""
    for model_dir in [obb_dir, pose_dir]:
        (model_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (model_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (model_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (model_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    splits = [("train", train_dirs), ("val", val_dirs)]

    for split_name, src_dirs in splits:
        total_count = 0
        for src_path in src_dirs:
            gt_file = src_path / "gt_annotations.json"
            if not gt_file.exists():
                continue

            with open(gt_file, encoding="utf-8") as f:
                gt = json.load(f)

            prefix = src_path.name  # train_front, val_back,...
            for rel_key, entry in gt.items():
                img_src = src_path / rel_key
                if not img_src.exists():
                    continue

                img = cv2.imread(str(img_src))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                clean_name = rel_key.replace("/", "__").replace("\\", "__")
                new_img_name = f"{prefix}__{clean_name}"
                txt_name = Path(new_img_name).stem + ".txt"

                # Copy image
                shutil.copy2(img_src, obb_dir / "images" / split_name / new_img_name)
                shutil.copy2(img_src, pose_dir / "images" / split_name / new_img_name)

                # Labels
                corners = entry["corners"]
                obb_txt = corners_to_obb_label(corners, img_w, img_h)
                pose_txt = corners_to_pose_label(corners, img_w, img_h)

                with open(obb_dir / "labels" / split_name / txt_name, "w", encoding="utf-8") as f:
                    f.write(obb_txt + "\n")

                with open(pose_dir / "labels" / split_name / txt_name, "w", encoding="utf-8") as f:
                    f.write(pose_txt + "\n")

                total_count += 1

        print(f"  ✓ {split_name.upper()} split: {total_count} ảnh")

    # Tạo dataset.yaml cho OBB
    obb_yaml = f"""path: {obb_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: cccd_card
"""
    with open(obb_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(obb_yaml)

    # Tạo dataset.yaml cho Pose
    pose_yaml = f"""path: {pose_dir.resolve().as_posix()}
train: images/train
val: images/val

kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]

names:
  0: cccd_card
"""
    with open(pose_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(pose_yaml)

    print(f"✅ YOLO-OBB dataset sẵn sàng tại: {obb_dir}")
    print(f"✅ YOLO-Pose dataset sẵn sàng tại: {pose_dir}")


def main():
    data_new = PROJECT_ROOT / "data_new"
    card_front = PROJECT_ROOT / "assets" / "samples" / "clean_front.jpg"
    card_back  = PROJECT_ROOT / "assets" / "samples" / "clean_back.jpg"

    if not card_front.exists() or not card_back.exists():
        print("❌ Không tìm thấy clean_front.jpg hoặc clean_back.jpg trong assets/samples/")
        return

    print("=======================================================")
    print("🚀 Bắt đầu sinh bộ dữ liệu mới tại data_new/")
    print("=======================================================\n")

    # 1. Sinh data cho Front (60 train, 20 val, 20 test per category)
    print("📷 [1/4] Sinh Synthetic Front...")
    generate_comprehensive_dataset(str(card_front), str(data_new / "train_front"), samples_per_category=60)
    generate_comprehensive_dataset(str(card_front), str(data_new / "val_front"), samples_per_category=20)
    generate_comprehensive_dataset(str(card_front), str(data_new / "synthetic_testset_front"), samples_per_category=20)

    # 2. Sinh data cho Back (60 train, 20 val, 20 test per category)
    print("\n📷 [2/4] Sinh Synthetic Back...")
    generate_comprehensive_dataset(str(card_back), str(data_new / "train_back"), samples_per_category=60)
    generate_comprehensive_dataset(str(card_back), str(data_new / "val_back"), samples_per_category=20)
    generate_comprehensive_dataset(str(card_back), str(data_new / "synthetic_testset_back"), samples_per_category=20)

    # 3. Copy Real Test
    print("\n📁 [3/4] Copy Real Test...")
    copy_real_test(PROJECT_ROOT / "data" / "real_test", data_new / "real_test")

    # 4. Convert YOLO
    print("\n🔄 [4/4] Convert nhãn sang YOLO-OBB & YOLO-Pose...")
    train_dirs = [data_new / "train_front", data_new / "train_back"]
    val_dirs   = [data_new / "val_front",   data_new / "val_back"]
    convert_dataset_to_yolo(train_dirs, val_dirs, data_new / "yolo_obb", data_new / "yolo_pose")

    print("\n=======================================================")
    print("🎉 TẠO BỘ DỮ LIỆU DATA_NEW HOÀN TẤT!")
    print("=======================================================")


if __name__ == "__main__":
    main()
