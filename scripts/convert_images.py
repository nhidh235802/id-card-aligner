"""
convert_images.py – Convert tất cả ảnh trong thư mục về .jpg để đảm bảo tương thích.

Hỗ trợ: .webp, .png, .bmp, .tiff, .jpeg → .jpg
Xóa file gốc sau khi convert thành công (có thể tắt bằng --keep).

Cách dùng:
    python scripts/convert_images.py --dir data/real_test
    python scripts/convert_images.py --dir data/real_test --keep   # giữ file gốc
"""

import cv2
import argparse
from pathlib import Path

SUPPORTED = {".webp", ".png", ".bmp", ".tiff", ".tif", ".jpeg"}


def convert_folder(folder: str, keep_originals: bool = False):
    folder_path = Path(folder)
    files = [f for f in folder_path.rglob("*") if f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"Không tìm thấy file nào cần convert trong: {folder}")
        return

    print(f"Tìm thấy {len(files)} file cần convert...")
    ok, fail = 0, 0

    for src in files:
        img = cv2.imread(str(src))
        if img is None:
            print(f"  ❌ Không đọc được: {src.name}")
            fail += 1
            continue

        dst = src.with_suffix(".jpg")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  ✅ {src.name}  →  {dst.name}")

        if not keep_originals:
            src.unlink()
        ok += 1

    print(f"\nHoàn tất: {ok} thành công | {fail} lỗi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  required=True, help="Thư mục chứa ảnh cần convert")
    parser.add_argument("--keep", action="store_true", help="Giữ lại file gốc sau khi convert")
    args = parser.parse_args()
    convert_folder(args.dir, args.keep)
