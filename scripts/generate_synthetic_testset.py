"""
generate_synthetic_testset.py – Tự động tạo Synthetic Test Set bao hàm kỹ các trường hợp ngách.

Phân loại 6 Sub-datasets:
1. easy_standard     : Xoay nhẹ (±15°), sáng rõ, nền đơn giản.
2. extreme_rotation  : Xoay 90°, 180°, 270° hoặc chéo góc nặng.
3. perspective_3d    : Nghiêng 3D biến dạng hình thang cực đại.
4. occlusion         : Ngón tay / vật thể che 1 hoặc 2 góc thẻ.
5. low_contrast      : Nền trùng màu thẻ (xanh/trắng), viền mờ.
6. lighting_blur     : Lóa sáng Flash hoặc mờ nhòe (motion blur).
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path


def rotate_image_and_corners(image, angle_deg):
    """Xoay ảnh và tính lại tọa độ 4 góc chính xác."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated_img = cv2.warpAffine(image, M, (new_w, new_h))
    
    # 4 góc ban đầu
    orig_corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)
    
    # Transform corners
    ones = np.ones(shape=(len(orig_corners), 1))
    points_ones = np.hstack([orig_corners, ones])
    new_corners = M.dot(points_ones.T).T
    
    return rotated_img, new_corners


def apply_perspective(image, corners, distortion_level=0.2):
    """Nghiêng 3D bằng Perspective Transform."""
    h, w = image.shape[:2]
    dx = w * distortion_level
    dy = h * distortion_level
    
    src_pts = corners.astype(np.float32)
    dst_pts = np.float32([
        [corners[0][0] + random.uniform(0, dx), corners[0][1] + random.uniform(0, dy)],
        [corners[1][0] - random.uniform(0, dx), corners[1][1] + random.uniform(0, dy)],
        [corners[2][0] - random.uniform(0, dx), corners[2][1] - random.uniform(0, dy)],
        [corners[3][0] + random.uniform(0, dx), corners[3][1] - random.uniform(0, dy)]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(image, M, (w, h))
    return warped_img, dst_pts


def add_finger_occlusion(image, corners, num_corners=1):
    """
    Giả lập ngón tay che góc thẻ.

    Cải thiện so với hình tròn đơn thuần:
    - Vẽ hình chữ nhật xéo hướng ra ngoài góc thẻ để che luôn 1 phần cạnh viền,
      mô phỏng ngón tay thực tế cầm thẻ gần sát góc.

    Known limitation (ghi chú báo cáo):
    - Mẫu ngón tay vẫn là hình học đơn giản, chưa mô phỏng kết cấu da thật.
    - Đủ dùng cho mục đích so sánh định tính/định lượng trong báo cáo.
    """
    vis = image.copy()
    h, w = image.shape[:2]
    canvas_center = np.array([w / 2, h / 2], dtype=np.float32)
    indices = random.sample(range(4), min(num_corners, 4))

    for idx in indices:
        pt = corners[idx].astype(np.float32)
        # Hướng từ tâm canvas ra góc (để hình chữ nhật che từ ngoài vào)
        direction = pt - canvas_center
        norm = np.linalg.norm(direction)
        unit_dir = direction / norm if norm > 0 else np.array([1.0, 0.0])

        finger_len = int(min(h, w) * random.uniform(0.18, 0.28))
        finger_width = int(min(h, w) * random.uniform(0.07, 0.12))
        # Màu da ngón tay (Skintone)
        color = (
            random.randint(110, 200),
            random.randint(120, 195),
            random.randint(130, 210),
        )

        # Tính 4 góc hình chữ nhật ngón tay
        perp = np.array([-unit_dir[1], unit_dir[0]])
        tip = pt + unit_dir * finger_len
        rect_pts = np.array([
            pt    + perp * finger_width,
            pt    - perp * finger_width,
            tip   - perp * finger_width,
            tip   + perp * finger_width,
        ], dtype=np.int32)

        cv2.fillPoly(vis, [rect_pts], color)
        # Thêm 1 hình tròn ở đầu ngón để tự nhiên hơn
        cv2.circle(vis, (int(tip[0]), int(tip[1])), finger_width, color, -1)

    return vis


def add_glare_flash(image):
    """Thêm vệt lóa ánh sáng Flash."""
    h, w = image.shape[:2]
    overlay = image.copy()
    center = (random.randint(0, w), random.randint(0, h))
    radius = random.randint(min(h, w) // 4, min(h, w) // 2)
    
    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    alpha = random.uniform(0.3, 0.6)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def generate_comprehensive_dataset(card_path: str, output_base: str, samples_per_category: int = 20):
    card_img = cv2.imread(card_path)
    if card_img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {card_path}")

    out_base = Path(output_base)
    categories = [
        "1_easy_standard",
        "2_extreme_rotation",
        "3_perspective_3d",
        "4_occlusion",
        "5_low_contrast",
        "6_lighting_blur"
    ]
    
    for cat in categories:
        (out_base / cat).mkdir(parents=True, exist_ok=True)

    all_gt = {}

    card_h, card_w = card_img.shape[:2]
    base_corners = np.array([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]], dtype=np.float32)

    for cat in categories:
        print(f"Đang sinh {samples_per_category} ảnh cho nhóm: {cat}...")
        
        for i in range(1, samples_per_category + 1):
            canvas_size = 900
            # Nền mặc định
            bg_color = random.randint(210, 240)
            bg = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
            
            curr_card = card_img.copy()
            curr_corners = base_corners.copy()

            # --- Biến đổi theo từng Category ---
            if cat == "1_easy_standard":
                angle = random.uniform(-15, 15)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)

            elif cat == "2_extreme_rotation":
                # Chọn các góc nghiêng nặng: 90, 180, 270 hoặc chéo 45, 135
                angle = random.choice([45, 90, 135, 180, 225, 270]) + random.uniform(-10, 10)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)

            elif cat == "3_perspective_3d":
                angle = random.uniform(-20, 20)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)
                curr_card, curr_corners = apply_perspective(curr_card, curr_corners, distortion_level=0.35)

            elif cat == "4_occlusion":
                angle = random.uniform(-30, 30)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)

            elif cat == "5_low_contrast":
                # Nền màu gần giống viền thẻ CCCD (xám/trắng/xanh nhạt)
                bg_color_choice = random.choice([
                    (210, 180, 140),   # Nền gỗ sáng
                    (220, 220, 220),   # Nền xám trắng
                    (180, 200, 210),   # Nền xanh nhạt giống viền CCCD
                ])
                bg = np.full((canvas_size, canvas_size, 3), bg_color_choice, dtype=np.uint8)
                angle = random.uniform(-25, 25)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)

                # BUG FIX: alpha-blend chỉ trên vùng thực sự là thẻ, không blend
                # vào phần canvas đen (padding của warpAffine) vì sau đó bước
                #   cv2.threshold(gray, 1, 255, THRESH_BINARY)
                # dùng "đen = trong suốt" để tạo mask dán thẻ — nếu vùng đen đã
                # bị blend thành màu xám/be thì mask sẽ nhận nhầm vùng đó là thẻ.
                #
                # Giải pháp: lưu mask vùng thẻ TRƯỚC khi blend,
                # sau khi blend xong restore lại các pixel ngoài thẻ về đen.
                _gray_pre = cv2.cvtColor(curr_card, cv2.COLOR_BGR2GRAY)
                _, _card_mask = cv2.threshold(_gray_pre, 1, 255, cv2.THRESH_BINARY)

                alpha = random.uniform(0.55, 0.80)
                bg_card = np.full(curr_card.shape, bg_color_choice, dtype=np.uint8)
                curr_card = cv2.addWeighted(curr_card, alpha, bg_card, 1 - alpha, 0)

                # Restore vùng ngoài thẻ về đen để threshold phía sau hoạt động đúng
                curr_card[_card_mask == 0] = 0

            elif cat == "6_lighting_blur":
                angle = random.uniform(-30, 30)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)
                if random.random() < 0.6:
                    curr_card = add_glare_flash(curr_card)

            # --- Dán thẻ vào Background Canvas ---
            ch, cw = curr_card.shape[:2]
            scale = min((canvas_size * 0.65) / cw, (canvas_size * 0.65) / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            
            resized_card = cv2.resize(curr_card, (new_w, new_h))
            scaled_corners = curr_corners * scale

            offset_x = (canvas_size - new_w) // 2 + random.randint(-40, 40)
            offset_y = (canvas_size - new_h) // 2 + random.randint(-40, 40)

            # Clamp offset để đảm bảo thẻ không vượt ra ngoài biên canvas
            # (tránh bug âm thầm: roi bị crop → final_corners lệch so với ảnh thật)
            offset_x = int(np.clip(offset_x, 0, canvas_size - new_w))
            offset_y = int(np.clip(offset_y, 0, canvas_size - new_h))

            final_corners = scaled_corners + np.array([offset_x, offset_y], dtype=np.float32)

            # Ghép vào nền
            gray_card = cv2.cvtColor(resized_card, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_card, 1, 255, cv2.THRESH_BINARY)
            
            roi = bg[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
            idx = (mask > 0)
            roi[idx] = resized_card[idx]

            # Xử lý hiệu ứng sau khi dán lên canvas.
            # NOTE: Các hiệu ứng này KHÔNG làm dịch chuyển vị trí 4 góc thẻ
            # nên final_corners vẫn chính xác — đây là thiết kế cố ý, không phải bug.
            if cat == "4_occlusion":
                num_occ = random.choice([1, 2])
                bg = add_finger_occlusion(bg, final_corners, num_corners=num_occ)

            if cat == "6_lighting_blur" and random.random() < 0.5:
                ksize = random.choice([7, 9, 11])  # Đa dạng mức độ mờ
                bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

            # Lưu file ảnh & GT
            file_name = f"{cat}_{i:03d}.jpg"
            save_path = out_base / cat / file_name
            cv2.imwrite(str(save_path), bg)

            all_gt[f"{cat}/{file_name}"] = {
                "corners": final_corners.tolist(),
                "category": cat,
                "is_occluded": (cat == "4_occlusion")
            }

    # Lưu file Annotation tổng
    with open(out_base / "gt_annotations.json", "w", encoding="utf-8") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Hoàn thành! Đã tạo tổng cộng {samples_per_category * len(categories)} ảnh test tại thư mục '{out_base}'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True, help="Đường dẫn đến 1 ảnh CCCD phẳng sạch")
    parser.add_argument("--output", default="data/synthetic_testset", help="Thư mục lưu dataset")
    parser.add_argument("--samples", type=int, default=20, help="Số lượng ảnh sinh ra cho MỖI category")
    args = parser.parse_args()

    generate_comprehensive_dataset(args.card, args.output, args.samples)
