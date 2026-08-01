"""
generate_synthetic_testset.py – Tự động tạo Synthetic Test Set từ vài ảnh sạch gốc.

Cách hoạt động:
1. Nhận ảnh CCCD phẳng đã crop sạch (hoặc tự động crop).
2. Dán ảnh CCCD lên các background thực tế ngẫu nhiên (nền bàn, nền sàn...).
3. Áp dụng các biến đổi 2D/3D (Rotate, Perspective Transform, Blur, Occlusion/Hand crop).
4. Tự động tính toán vị trí Ground Truth 4 góc [TL, TR, BR, BL] và ghi ra file annotation JSON.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path


def random_perspective_transform(img, max_ratio=0.25):
    """Mô phỏng góc nghiêng 3D bằng Perspective Transform."""
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Dịch chuyển ngẫu nhiên 4 góc
    dx = w * max_ratio
    dy = h * max_ratio
    
    dst_pts = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), h - random.uniform(0, dy)],
        [random.uniform(0, dx), h - random.uniform(0, dy)]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, dst_pts


def add_occlusion(img, corners):
    """Mô phỏng ngón tay hoặc vật thể che 1 góc thẻ."""
    h, w = img.shape[:2]
    # Chọn ngẫu nhiên 1 trong 4 góc để che
    idx = random.randint(0, 3)
    pt = corners[idx]
    
    # Vẽ 1 ellipse/polygon giả làm ngón tay che góc
    radius = int(min(h, w) * random.uniform(0.08, 0.15))
    cv2.circle(img, (int(pt[0]), int(pt[1])), radius, (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)), -1)
    return img


def generate_synthetic_samples(input_card_path: str, output_dir: str, num_samples: int = 20):
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    card_img = cv2.imread(input_card_path)
    if card_img is None:
        print(f"Lỗi: Không tìm thấy ảnh {input_card_path}")
        return
        
    card_h, card_w = card_img.shape[:2]
    gt_annotations = {}

    for i in range(num_samples):
        # 1. Khởi tạo canvas nền (giả lập background 800x800)
        canvas_size = 800
        bg = np.full((canvas_size, canvas_size, 3), random.randint(200, 245), dtype=np.uint8)
        
        # Add random noise/texture to bg
        noise = np.random.randint(-20, 20, (canvas_size, canvas_size, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 2. Resize card vừa vặn trong canvas
        scale = random.uniform(0.4, 0.6) * (canvas_size / max(card_h, card_w))
        new_w, new_h = int(card_w * scale), int(card_h * scale)
        resized_card = cv2.resize(card_img, (new_w, new_h))

        # 3. Tạo transform nghiêng + xoay
        M, corners = random_perspective_transform(resized_card, max_ratio=0.2)

        # 4. Đặt card vào giữa canvas
        offset_x = (canvas_size - new_w) // 2 + random.randint(-50, 50)
        offset_y = (canvas_size - new_h) // 2 + random.randint(-50, 50)

        # Warp card
        warped_card = cv2.warpPerspective(resized_card, M, (new_w, new_h))
        
        # Mask cho card
        mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, M, (new_w, new_h))

        # Ghép card vào bg tại offset
        roi = bg[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        idx = (warped_mask > 0)
        roi[idx] = warped_card[idx]

        # 5. Cập nhật Ground Truth Corners theo vị trí mới trên canvas
        final_corners = corners + np.array([offset_x, offset_y], dtype=np.float32)

        # 6. Thêm nhiễu ngẫu nhiên: Occlusion, Blur, Brightness
        if random.random() < 0.3:
            bg = add_occlusion(bg, final_corners)
        
        if random.random() < 0.4:
            ksize = random.choice([3, 5, 7])
            bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

        # Save ảnh & ground truth
        img_name = f"synth_{Path(input_card_path).stem}_{i:03d}.jpg"
        cv2.imwrite(str(images_dir / img_name), bg)
        gt_annotations[img_name] = final_corners.tolist()

    # Save JSON Ground Truth
    with open(out_dir / "gt_annotations.json", "w") as f:
        json.dump(gt_annotations, f, indent=2)

    print(f"Đã sinh {num_samples} ảnh test synthetic tại: {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True, help="Đường dẫn đến 1 ảnh CCCD phẳng gốc")
    parser.add_argument("--output", default="data/splits/synthetic_test", help="Thư mục xuất kết quả")
    parser.add_argument("--num", type=int, default=20, help="Số lượng ảnh tạo ra")
    args = parser.parse_args()

    generate_synthetic_samples(args.card, args.output, args.num)
