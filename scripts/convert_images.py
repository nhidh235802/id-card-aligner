import cv2
import argparse
from pathlib import Path

# Các định dạng định dạng ảnh hỗ trợ chuyển đổi về .jpg
SUPPORTED = {".webp", ".png", ".bmp", ".tiff", ".tif", ".jpeg"}


def convert_folder(folder: str, keep_originals: bool = False):
    """Quét và chuyển đổi toàn bộ ảnh có định dạng trong SUPPORTED về định dạng .jpg."""
    folder_path = Path(folder)
    # Lấy danh sách tất cả các file có đuôi mở rộng phù hợp
    files = [f for f in folder_path.rglob("*") if f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"Không tìm thấy file nào cần convert trong: {folder}")
        return

    print(f"Tìm thấy {len(files)} file cần convert...")
    ok, fail = 0, 0

    # Lặp qua từng file ảnh để đọc và ghi lại với chuẩn JPG
    for src in files:
        img = cv2.imread(str(src))
        if img is None:
            print(f"  ❌ Không đọc được: {src.name}")
            fail += 1
            continue

        # Đặt tên file đầu ra có phần mở rộng .jpg
        dst = src.with_suffix(".jpg")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  ✅ {src.name}  →  {dst.name}")

        # Xóa file nguồn nếu người dùng không yêu cầu giữ lại (--keep)
        if not keep_originals:
            src.unlink()
        ok += 1

    print(f"\nHoàn tất: {ok} thành công | {fail} lỗi")


if __name__ == "__main__":
    # Khởi tạo parser đọc các đối số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  required=True, help="Thư mục chứa ảnh cần convert")
    parser.add_argument("--keep", action="store_true", help="Giữ lại file gốc sau khi convert")
    args = parser.parse_args()
    
    convert_folder(args.dir, args.keep)
