import cv2
import json
import argparse
import random
import numpy as np
from pathlib import Path

# Các nhãn và màu sắc định nghĩa cho 4 góc thẻ [TL, TR, BR, BL]
CORNER_LABELS  = ["TL", "TR", "BR", "BL"]
CORNER_COLORS  = [(0, 255, 0), (255, 100, 0), (0, 0, 255), (0, 220, 220)]


def draw_gt_overlay(image: np.ndarray, corners: list, category: str) -> np.ndarray:
    """Vẽ đa giác Ground Truth 4 góc đè lên ảnh gốc để hỗ trợ kiểm tra mắt thường."""
    vis = image.copy()
    pts = np.array(corners, dtype=np.float32)

    # ── Khối 1: Vẽ khung polygon viền xanh lá thẻ ──
    cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)

    # ── Khối 2: Vẽ từng điểm góc tròn có viền trắng kèm tên nhãn ──
    for i, (pt, label, color) in enumerate(zip(pts, CORNER_LABELS, CORNER_COLORS)):
        ix, iy = int(pt[0]), int(pt[1])
        cv2.circle(vis, (ix, iy), 8, color, -1)
        cv2.circle(vis, (ix, iy), 8, (255, 255, 255), 2)  # Viền màu trắng
        cv2.putText(vis, label, (ix + 10, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Khối 3: Vẽ thông tin phân loại (Category) lên góc trên ảnh ──
    cv2.putText(vis, f"GT | {category}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(vis, f"GT | {category}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1)

    return vis


def main():
    """Hàm điều khiển kiểm tra trực quan Ground Truth các ảnh ngẫu nhiên trong dataset."""
    parser = argparse.ArgumentParser(description="Visualize Ground Truth corners lên ảnh")
    parser.add_argument("--data",     required=True,  help="Thư mục chứa dataset tổng (có gt_annotations.json)")
    parser.add_argument("--category", default=None,   help="Lọc theo 1 category (vd: 4_occlusion)")
    parser.add_argument("--num",      type=int, default=10, help="Số ảnh lấy mẫu để kiểm tra (mặc định 10)")
    parser.add_argument("--save",     action="store_true",  help="Lưu ảnh overlay ra thư mục verify_output/")
    args = parser.parse_args()

    data_dir = Path(args.data)
    gt_path  = data_dir / "gt_annotations.json"

    # Kiểm tra sự tồn tại của file nhãn Ground Truth
    if not gt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file GT: {gt_path}")

    with open(gt_path, encoding="utf-8") as f:
        all_gt = json.load(f)

    # Lọc danh sách theo category nếu người dùng chỉ định
    keys = list(all_gt.keys())
    if args.category:
        keys = [k for k in keys if args.category in k]
        if not keys:
            print(f"Không tìm thấy ảnh nào thuộc category: {args.category}")
            return

    # Trích xuất mẫu ảnh ngẫu nhiên
    sample_keys = random.sample(keys, min(args.num, len(keys)))

    save_dir = data_dir / "verify_output"
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0

    print(f"\n{'─'*55}")
    print(f"  Kiểm tra {len(sample_keys)} ảnh GT ngẫu nhiên")
    print(f"{'─'*55}")

    # Lặp qua từng file trong danh sách mẫu ngẫu nhiên
    for key in sample_keys:
        img_path = data_dir / key
        if not img_path.exists():
            print(f"  ⚠️  Không tìm thấy ảnh: {img_path}")
            failed += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Lỗi đọc ảnh: {img_path}")
            failed += 1
            continue

        entry    = all_gt[key]
        corners  = entry["corners"]
        category = entry.get("category", "unknown")

        # ── Kiểm tra hợp lệ tự động: 4 góc có nằm hoàn toàn trong kích thước ảnh không ──
        h, w = img.shape[:2]
        pts  = np.array(corners, dtype=np.float32)
        in_bounds = np.all((pts[:, 0] >= 0) & (pts[:, 0] <= w) &
                           (pts[:, 1] >= 0) & (pts[:, 1] <= h))

        status = "✅ OK" if in_bounds else "❌ OUT-OF-BOUNDS"
        print(f"  {status}  {key}")
        if not in_bounds:
            failed += 1
        else:
            passed += 1

        # Tạo ảnh hiển thị trực quan có vẽ đè thông tin GT
        vis = draw_gt_overlay(img, corners, category)

        # Lưu ảnh hoặc hiển thị lên màn hình tùy chọn
        if args.save:
            out_name = key.replace("/", "__")
            cv2.imwrite(str(save_dir / out_name), vis)
        else:
            max_dim = 700
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                vis   = cv2.resize(vis, (int(w * scale), int(h * scale)))
            cv2.imshow(f"GT Verify — {category}", vis)
            key_press = cv2.waitKey(0)
            if key_press == ord("q"):
                break

    cv2.destroyAllWindows()

    print(f"\n{'─'*55}")
    print(f"  Tổng kết: ✅ {passed} ảnh OK | ❌ {failed} ảnh lỗi")
    if args.save:
        print(f"  Ảnh overlay đã lưu tại: {save_dir}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
