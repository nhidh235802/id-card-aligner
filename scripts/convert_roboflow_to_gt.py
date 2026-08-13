import cv2
import json
import shutil
import argparse
import numpy as np
from pathlib import Path


def convert(src_dir: str, out_dir: str):
    """Chuyển đổi nhãn Roboflow YOLO Pose xuất ra sang định dạng gt_annotations.json chuẩn của dự án."""
    src      = Path(src_dir)
    img_dir  = src / "images"
    lbl_dir  = src / "labels"
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_gt   = {}
    n_ok     = 0
    n_skip   = 0

    # Lấy danh sách tất cả các file nhãn Roboflow (.txt)
    label_files = sorted(lbl_dir.glob("*.txt"))
    print(f"Tìm thấy {len(label_files)} file label...")

    for lbl_path in label_files:
        # Tìm ảnh nguồn tương ứng với file label
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

        # Đọc kích thước ảnh để tính toán giải chuẩn hóa (denormalize) tọa độ
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Không đọc được ảnh: {img_path.name}")
            n_skip += 1
            continue
        img_h, img_w = img.shape[:2]

        # Đọc nội dung file nhãn TXT từ Roboflow
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            print(f"  ⚠️  File label rỗng: {lbl_path.name}")
            n_skip += 1
            continue

        # Trích xuất thông số từ dòng đầu tiên
        parts = list(map(float, lines[0].split()))

        # Kiểm tra độ dài hợp lệ của chuỗi nhãn Roboflow (tối thiểu 17 giá trị cho 4 keypoints)
        if len(parts) < 17:
            print(f"  ⚠️  Label thiếu dữ liệu ({len(parts)} values): {lbl_path.name}")
            n_skip += 1
            continue

        # Trích xuất 4 keypoints góc và trạng thái hiển thị (visibility)
        kpts = []
        visibilities = []
        for i in range(4):
            x_norm = parts[5 + i * 3]
            y_norm = parts[6 + i * 3]
            v      = int(parts[7 + i * 3])
            kpts.append([x_norm * img_w, y_norm * img_h])
            visibilities.append(v)

        # Bỏ qua nếu tất cả keypoints đều có visibility = 0
        if all(v == 0 for v in visibilities):
            print(f"  ⚠️  Tất cả keypoints visibility=0: {lbl_path.name}")
            n_skip += 1
            continue

        # Xác định trạng thái che khuất (v = 1 đại diện cho occluded)
        is_occluded = any(v == 1 for v in visibilities)

        # Chuẩn hóa lại tên file ảnh (loại bỏ hậu tố ngẫu nhiên của Roboflow)
        clean_name = lbl_path.stem.split("_jpg.rf.")[0] \
                              .split("_jpeg.rf.")[0] \
                              .split("_png.rf.")[0] + ".jpg"
        dst_img = out / clean_name

        # Xử lý tránh ghi đè nếu trùng tên file
        counter = 1
        while dst_img.exists():
            stem = clean_name.replace(".jpg", "")
            dst_img = out / f"{stem}_{counter}.jpg"
            counter += 1

        # Sao chép ảnh sang thư mục đích
        shutil.copy2(img_path, dst_img)

        # Ghi nhận nhãn định dạng GT của dự án
        all_gt[dst_img.name] = {
            "corners":     kpts,
            "category":    "real",
            "is_occluded": is_occluded,
            "visibility":  visibilities,
        }
        n_ok += 1
        print(f"  ✅ {dst_img.name}  | occluded={is_occluded} | vis={visibilities}")

    # Ghi toàn bộ dữ liệu GT ra file gt_annotations.json
    gt_path = out / "gt_annotations.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*55}")
    print(f"  ✅ Thành công: {n_ok} ảnh")
    print(f"  ⚠️  Bỏ qua:    {n_skip} ảnh")
    print(f"  GT lưu tại:  {gt_path}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/real_test_roboflow/test",
                        help="Thư mục Roboflow export (chứa images/ và labels/)")
    parser.add_argument("--out", default="data/real_test",
                        help="Thư mục output (chứa ảnh + gt_annotations.json)")
    args = parser.parse_args()
    convert(args.src, args.out)
