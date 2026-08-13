import json
import shutil
import random
import argparse
import sys
import numpy as np
from pathlib import Path
import cv2

# Đảm bảo PYTHONPATH trỏ đúng vào thư mục gốc của dự án
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.corner_utils import order_corners


# ── Hàm hỗ trợ tính toán tọa độ và định dạng nhãn ──────────────────────────────

def corners_to_bbox(corners: list, img_w: int, img_h: int):
    """Tính toán bounding box chuẩn hóa (cx, cy, w, h) từ 4 góc pixel."""
    pts = np.array(corners, dtype=np.float32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return cx, cy, w, h


def corners_to_obb_label(corners: list, img_w: int, img_h: int) -> str:
    """Chuyển đổi 4 góc pixel sang định dạng YOLO-OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4 (chuẩn hóa [0,1])."""
    pts = np.array(corners, dtype=np.float32)  # [TL, TR, BR, BL]
    normalized = pts / np.array([img_w, img_h], dtype=np.float32)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
    return f"0 {coords}"


def corners_to_pose_label(corners: list, img_w: int, img_h: int) -> str:
    """Chuyển đổi 4 góc pixel sang định dạng YOLO-Pose: class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4."""
    pts = np.array(corners, dtype=np.float32)  # (4,2) [TL,TR,BR,BL]
    cx, cy, bw, bh = corners_to_bbox(pts.tolist(), img_w, img_h)
    normalized = pts / np.array([img_w, img_h], dtype=np.float32)
    kpt_str = "  ".join(f"{x:.6f} {y:.6f} 2" for x, y in normalized)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {kpt_str}"


def get_image_size(img_path: Path):
    """Đọc kích thước chiều rộng (w) và chiều cao (h) của ảnh từ đường dẫn file."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 900, 900  # Kích thước mặc định phòng trường hợp không đọc được ảnh
    return img.shape[1], img.shape[0]  # w, h


# ── Hàm xử lý chuyển đổi chính ──────────────────────────────────────────────────

def convert_sources(src_dirs: list, obb_out: str, pose_out: str, val_ratio: float):
    """Chuyển đổi toàn bộ nhãn GT trong các thư mục nguồn sang tập dữ liệu YOLO-OBB và YOLO-Pose."""
    all_items = []  # Danh sách lưu các phần tử (img_path, corners, is_occluded)

    # Đọc và gom tất cả ảnh từ các thư mục nguồn có gt_annotations.json
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

    # Xáo trộn và phân chia tập dữ liệu thành train / val theo tỷ lệ
    random.shuffle(all_items)
    n_val   = int(len(all_items) * val_ratio)
    val_items   = all_items[n_val:]
    train_items = all_items[:n_val]
    print(f"  Train: {len(train_items)} | Val: {len(val_items)}")

    # Tạo thư mục và xuất dữ liệu cho từng task (OBB và Pose)
    for task, out_dir in [("obb", Path(obb_out)), ("pose", Path(pose_out))]:
        for split, items in [("train", train_items), ("val", val_items)]:
            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, corners, is_occluded in items:
                # Đặt tên file duy nhất để tránh trùng lập giữa các bộ nguồn
                src_name = img_path.parent.parent.name  # Ví dụ: "train_front"
                new_stem = f"{src_name}__{img_path.parent.name}__{img_path.stem}"
                dst_img  = img_dir / f"{new_stem}.jpg"
                dst_lbl  = lbl_dir / f"{new_stem}.txt"

                # Sao chép file ảnh
                shutil.copy2(img_path, dst_img)

                img_w, img_h = get_image_size(img_path)

                # Chuyển đổi định dạng nhãn theo loại mô hình (OBB hoặc Pose)
                if task == "obb":
                    label_line = corners_to_obb_label(corners, img_w, img_h)
                else:
                    label_line = corners_to_pose_label(corners, img_w, img_h)

                dst_lbl.write_text(label_line + "\n", encoding="utf-8")

        # Tạo file cấu hình dataset.yaml cho YOLO
        if task == "pose":
            kpt_cfg = (
                "\nkpt_shape: [4, 3]  # 4 keypoints, mỗi kpt có (x, y, visibility)"
                "\nflip_idx: [1, 0, 3, 2]  # Khi flip ngang: TL↔TR (0↔1), BL↔BR (3↔2)"
            )
        else:
            kpt_cfg = ""
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
    # Đọc tham số dòng lệnh
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
