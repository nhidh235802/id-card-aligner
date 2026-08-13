import cv2
import numpy as np
from pathlib import Path

# Hằng số kích thước và tỷ lệ khung hình tiêu chuẩn ISO/IEC 7810 ID-1
TARGET_WIDTH = 856
TARGET_HEIGHT = 540
STANDARD_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # ≈ 1.585185


def sanitize_and_resize_card(input_path: str, output_path: str):
    """Chuẩn hóa một ảnh thẻ căn cước đầu vào về kích thước chuẩn ISO ID-1 (856x540 px)."""
    # Đọc ảnh nguồn
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh từ: {input_path}")

    h, w = img.shape[:2]
    curr_ratio = w / h

    print(f"Ảnh gốc: {w}x{h}px | Aspect Ratio hiện tại: {curr_ratio:.4f}")

    # Resize ảnh trực tiếp về kích thước chuẩn 856x540 dùng thuật toán Interpolation INTER_AREA
    clean_card = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Đảm bảo thư mục lưu tồn tại và ghi file ảnh đã chuẩn hóa
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), clean_card)

    print(f"✅ Đã chuẩn hóa ảnh CCCD sạch về kích thước chuẩn ISO ({TARGET_WIDTH}x{TARGET_HEIGHT}px, Ratio: {STANDARD_ASPECT_RATIO:.4f})")
    print(f"📁 Đã lưu tại: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chuẩn hóa ảnh CCCD gốc")
    parser.add_argument("--input", required=True, help="Đường dẫn ảnh CCCD nhặt trên mạng")
    parser.add_argument("--output", default="assets/samples/clean_cccd_standard.jpg", help="Nơi lưu ảnh sạch chuẩn")
    args = parser.parse_args()

    sanitize_and_resize_card(args.input, args.output)
