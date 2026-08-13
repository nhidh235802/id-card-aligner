import os
import sys
import shutil
import json
from pathlib import Path
import numpy as np
import cv2

# Thêm thư mục gốc vào sys.path để import các module nội bộ
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_testset import generate_comprehensive_dataset
from scripts.convert_gt_to_yolo import corners_to_obb_label, corners_to_pose_label, corners_to_bbox


def copy_real_test(src_real: Path, dst_real: Path):
    """Sao chép bộ ảnh kiểm thử thực tế và file nhãn gt_annotations.json sang data_new/real_test/."""
    dst_real.mkdir(parents=True, exist_ok=True)

    # Kiểm tra sự tồn tại của thư mục nguồn
    if not src_real.exists():
        print(f"⚠️  Thư mục nguồn '{src_real}' không tồn tại!")
        return

    # Kiểm tra sự tồn tại của file nhãn Ground Truth
    gt_json = src_real / "gt_annotations.json"
    if not gt_json.exists():
        print(f"⚠️  Không tìm thấy '{gt_json}'!")
        return

    # Đọc dữ liệu nhãn Ground Truth JSON
    with open(gt_json, encoding="utf-8") as f:
        gt_data = json.load(f)

    # Sao chép file gt_annotations.json sang thư mục đích
    shutil.copy2(gt_json, dst_real / "gt_annotations.json")

    # Duyệt và sao chép từng file ảnh được liệt kê trong file nhãn
    count = 0
    for rel_key in gt_data.keys():
        src_img = src_real / rel_key
        if src_img.exists():
            shutil.copy2(src_img, dst_real / rel_key)
            count += 1
        else:
            print(f"  ⚠️  Không tìm thấy ảnh: {rel_key}")

    print(f"✅ Đã copy chính xác {count} ảnh real test được gán nhãn + gt_annotations.json sang '{dst_real}'")


def convert_dataset_to_yolo(train_dirs: list, val_dirs: list, obb_dir: Path, pose_dir: Path):
    """Chuyển đổi dữ liệu tổng hợp trong train_dirs và val_dirs sang định dạng YOLO-OBB và YOLO-Pose."""
    # Tạo cấu trúc thư mục chuẩn cho YOLO (images/train, images/val, labels/train, labels/val)
    for model_dir in [obb_dir, pose_dir]:
        (model_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (model_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (model_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (model_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    splits = [("train", train_dirs), ("val", val_dirs)]

    # Duyệt qua các tập split (train, val) để chuyển đổi dữ liệu
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

                # Đọc ảnh để lấy kích thước chiều rộng và chiều cao
                img = cv2.imread(str(img_src))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                clean_name = rel_key.replace("/", "__").replace("\\", "__")
                new_img_name = f"{prefix}__{clean_name}"
                txt_name = Path(new_img_name).stem + ".txt"

                # Sao chép file ảnh sang thư mục hình ảnh của YOLO
                shutil.copy2(img_src, obb_dir / "images" / split_name / new_img_name)
                shutil.copy2(img_src, pose_dir / "images" / split_name / new_img_name)

                # Chuyển đổi tọa độ 4 góc sang định dạng label OBB và Pose
                corners = entry["corners"]
                obb_txt = corners_to_obb_label(corners, img_w, img_h)
                pose_txt = corners_to_pose_label(corners, img_w, img_h)

                # Ghi file nhãn .txt cho YOLO-OBB và YOLO-Pose
                with open(obb_dir / "labels" / split_name / txt_name, "w", encoding="utf-8") as f:
                    f.write(obb_txt + "\n")

                with open(pose_dir / "labels" / split_name / txt_name, "w", encoding="utf-8") as f:
                    f.write(pose_txt + "\n")

                total_count += 1

        print(f"  ✓ {split_name.upper()} split: {total_count} ảnh")

    # Ghi file cấu hình dataset.yaml cho YOLO-OBB
    obb_yaml = f"""path: {obb_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: cccd_card
"""
    with open(obb_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(obb_yaml)

    # Ghi file cấu hình dataset.yaml cho YOLO-Pose
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
    """Hàm điều khiển quy trình khởi tạo bộ dữ liệu data_new hoàn chỉnh."""
    data_new = PROJECT_ROOT / "data_new"
    card_front = PROJECT_ROOT / "assets" / "samples" / "clean_front.jpg"
    card_back  = PROJECT_ROOT / "assets" / "samples" / "clean_back.jpg"

    # Đảm bảo mẫu ảnh thẻ mặt trước và mặt sau tồn tại
    if not card_front.exists() or not card_back.exists():
        print("❌ Không tìm thấy clean_front.jpg hoặc clean_back.jpg trong assets/samples/")
        return

    print("=======================================================")
    print("🚀 Bắt đầu sinh bộ dữ liệu mới tại data_new/")
    print("=======================================================\n")

    # Bước 1: Sinh bộ dữ liệu tổng hợp mặt trước (train, val, test)
    print("📷 [1/4] Sinh Synthetic Front...")
    generate_comprehensive_dataset(str(card_front), str(data_new / "train_front"), samples_per_category=60)
    generate_comprehensive_dataset(str(card_front), str(data_new / "val_front"), samples_per_category=20)
    generate_comprehensive_dataset(str(card_front), str(data_new / "synthetic_testset_front"), samples_per_category=20)

    # Bước 2: Sinh bộ dữ liệu tổng hợp mặt sau (train, val, test)
    print("\n📷 [2/4] Sinh Synthetic Back...")
    generate_comprehensive_dataset(str(card_back), str(data_new / "train_back"), samples_per_category=60)
    generate_comprehensive_dataset(str(card_back), str(data_new / "val_back"), samples_per_category=20)
    generate_comprehensive_dataset(str(card_back), str(data_new / "synthetic_testset_back"), samples_per_category=20)

    # Bước 3: Sao chép tập ảnh thực tế làm testset
    print("\n📁 [3/4] Copy Real Test...")
    copy_real_test(PROJECT_ROOT / "data" / "real_test", data_new / "real_test")

    # Bước 4: Chuyển đổi nhãn sang định dạng huấn luyện YOLO-OBB và YOLO-Pose
    print("\n🔄 [4/4] Convert nhãn sang YOLO-OBB & YOLO-Pose...")
    train_dirs = [data_new / "train_front", data_new / "train_back"]
    val_dirs   = [data_new / "val_front",   data_new / "val_back"]
    convert_dataset_to_yolo(train_dirs, val_dirs, data_new / "yolo_obb", data_new / "yolo_pose")

    print("\n=======================================================")
    print("🎉 TẠO BỘ DỮ LIỆU DATA_NEW HOÀN TẤT!")
    print("=======================================================")


if __name__ == "__main__":
    main()
