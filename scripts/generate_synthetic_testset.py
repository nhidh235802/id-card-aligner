import cv2
import numpy as np
import json
import random
from pathlib import Path


def rotate_image_and_corners(image, angle_deg):
    """Xoay ảnh một góc angle_deg (độ) và tính toán lại chính xác vị trí tọa độ 4 góc thẻ."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # ── Khối 1: Tính ma trận xoay 2D và khung ảnh mở rộng ──
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated_img = cv2.warpAffine(image, M, (new_w, new_h))
    
    # ── Khối 2: Tính toán ma trận biến đổi vị trí 4 điểm góc ──
    orig_corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)
    
    ones = np.ones(shape=(len(orig_corners), 1))
    points_ones = np.hstack([orig_corners, ones])
    new_corners = M.dot(points_ones.T).T
    
    return rotated_img, new_corners


def apply_perspective(image, corners, distortion_level=0.2):
    """Giả lập độ nghiêng không gian 3D bằng Perspective Transform."""
    h, w = image.shape[:2]
    dx = w * distortion_level
    dy = h * distortion_level
    
    # Tạo các điểm biến dạng ngẫu nhiên ở 4 góc
    src_pts = corners.astype(np.float32)
    dst_pts = np.float32([
        [corners[0][0] + random.uniform(0, dx), corners[0][1] + random.uniform(0, dy)],
        [corners[1][0] - random.uniform(0, dx), corners[1][1] + random.uniform(0, dy)],
        [corners[2][0] - random.uniform(0, dx), corners[2][1] - random.uniform(0, dy)],
        [corners[3][0] + random.uniform(0, dx), corners[3][1] - random.uniform(0, dy)]
    ])
    
    # Áp dụng ma trận biến đổi góc nhìn 3D
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(image, M, (w, h))
    return warped_img, dst_pts


def add_finger_occlusion(image, corners, num_corners=1):
    """Vẽ giả lập hình ngón tay che đè trực tiếp lên góc thẻ để tạo case occlusion."""
    vis = image.copy()
    h, w = image.shape[:2]
    canvas_center = np.array([w / 2, h / 2], dtype=np.float32)
    indices = random.sample(range(4), min(num_corners, 4))

    for idx in indices:
        pt = corners[idx].astype(np.float32)
        # Vector hướng từ góc thẻ vào trong tâm canvas
        inward_dir = canvas_center - pt
        norm = np.linalg.norm(inward_dir)
        unit_inward = inward_dir / norm if norm > 0 else np.array([-1.0, 0.0])

        finger_width = int(min(h, w) * random.uniform(0.06, 0.11))
        overlap = random.uniform(35, 75)   # Độ sâu ngón tay đi vào góc thẻ
        finger_len = int(min(h, w) * random.uniform(0.20, 0.35))

        base_start = pt - unit_inward * (finger_len - overlap)
        tip = pt + unit_inward * overlap

        perp = np.array([-unit_inward[1], unit_inward[0]])
        color = (
            random.randint(120, 190),
            random.randint(140, 210),
            random.randint(170, 235),
        )

        rect_pts = np.array([
            base_start + perp * finger_width,
            base_start - perp * finger_width,
            tip        - perp * finger_width,
            tip        + perp * finger_width,
        ], dtype=np.int32)

        # Vẽ hình chữ nhật và đầu hình tròn của ngón tay
        cv2.fillPoly(vis, [rect_pts], color)
        cv2.circle(vis, (int(tip[0]), int(tip[1])), finger_width, color, -1)

    return vis


def create_realistic_background(canvas_size=900):
    """Sinh ảnh nền ngẫu nhiên phong phú: mặt bàn gỗ, bàn xám, gradient ánh sáng."""
    bg_type = random.choice(["wood", "desk", "gray_table", "textured_light"])

    # ── Tạo chất liệu nền gỗ ──
    if bg_type == "wood":
        base_c = np.array([random.randint(40, 90), random.randint(70, 130), random.randint(120, 180)], dtype=np.float32)
        bg = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8) + base_c
        noise = np.random.randint(-25, 25, (canvas_size, 1, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (5, 35), 0)

    # ── Tạo mặt bàn đơn sắc kèm nhiễu hạt ──
    elif bg_type == "desk":
        val = random.randint(180, 230)
        bg = np.full((canvas_size, canvas_size, 3), val, dtype=np.uint8)
        noise = np.random.normal(0, 12, (canvas_size, canvas_size, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Tạo nền chuyển sắc gradient ──
    else:
        val_start = random.randint(180, 230)
        val_end = random.randint(140, 190)
        X, Y = np.meshgrid(np.linspace(0, 1, canvas_size), np.linspace(0, 1, canvas_size))
        grad = (val_start * (1 - X) + val_end * X).astype(np.uint8)
        bg = cv2.merge([grad, grad, grad])

    # Thêm hiệu ứng viền tối vignette nhẹ xung quanh canvas
    Y, X = np.ogrid[:canvas_size, :canvas_size]
    dist_from_center = np.sqrt((X - canvas_size/2)**2 + (Y - canvas_size/2)**2)
    max_dist = np.sqrt(2) * (canvas_size/2)
    vig = 1.0 - 0.25 * (dist_from_center / max_dist)
    return (bg.astype(np.float32) * vig[:, :, np.newaxis]).astype(np.uint8)


def add_glare_flash(image):
    """Tạo hiệu ứng vệt ánh sáng chói Flash phủ lên ảnh thẻ."""
    h, w = image.shape[:2]
    overlay = image.copy()
    center = (random.randint(0, w), random.randint(0, h))
    radius = random.randint(min(h, w) // 4, min(h, w) // 2)

    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    alpha = random.uniform(0.3, 0.6)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def generate_comprehensive_dataset(card_path: str, output_base: str, samples_per_category: int = 20):
    """Quy trình tự động sinh bộ dữ liệu synthetic tổng hợp bao hàm 6 kịch bản môi trường thực tế."""
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
            bg = create_realistic_background(canvas_size)
            
            curr_card = card_img.copy()
            curr_corners = base_corners.copy()

            # ── Áp dụng hiệu ứng theo từng nhóm Category ──
            if cat == "1_easy_standard":
                angle = random.uniform(-15, 15)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)

            elif cat == "2_extreme_rotation":
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
                bg_color_choice = random.choice([
                    (210, 180, 140),
                    (220, 220, 220),
                    (180, 200, 210),
                ])
                bg = np.full((canvas_size, canvas_size, 3), bg_color_choice, dtype=np.uint8)
                angle = random.uniform(-25, 25)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)

                _gray_pre = cv2.cvtColor(curr_card, cv2.COLOR_BGR2GRAY)
                _, _card_mask = cv2.threshold(_gray_pre, 1, 255, cv2.THRESH_BINARY)

                alpha = random.uniform(0.55, 0.80)
                bg_card = np.full(curr_card.shape, bg_color_choice, dtype=np.uint8)
                curr_card = cv2.addWeighted(curr_card, alpha, bg_card, 1 - alpha, 0)
                curr_card[_card_mask == 0] = 0

            elif cat == "6_lighting_blur":
                angle = random.uniform(-30, 30)
                curr_card, curr_corners = rotate_image_and_corners(curr_card, angle)
                if random.random() < 0.6:
                    curr_card = add_glare_flash(curr_card)

            # ── Khối căn tỷ lệ và dán thẻ lên Canvas nền ──
            ch, cw = curr_card.shape[:2]
            scale = min((canvas_size * 0.65) / cw, (canvas_size * 0.65) / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            
            resized_card = cv2.resize(curr_card, (new_w, new_h))
            scaled_corners = curr_corners * scale

            offset_x = (canvas_size - new_w) // 2 + random.randint(-40, 40)
            offset_y = (canvas_size - new_h) // 2 + random.randint(-40, 40)

            offset_x = int(np.clip(offset_x, 0, canvas_size - new_w))
            offset_y = int(np.clip(offset_y, 0, canvas_size - new_h))

            final_corners = scaled_corners + np.array([offset_x, offset_y], dtype=np.float32)

            # Ghép thẻ vào nền dùng mask binary
            gray_card = cv2.cvtColor(resized_card, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_card, 1, 255, cv2.THRESH_BINARY)
            
            roi = bg[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
            idx = (mask > 0)
            roi[idx] = resized_card[idx]

            # ── Thêm hiệu ứng hậu xử lý (ngón tay che / mờ nhòe) ──
            if cat == "4_occlusion":
                num_occ = random.choice([1, 2])
                bg = add_finger_occlusion(bg, final_corners, num_corners=num_occ)

            if cat == "6_lighting_blur" and random.random() < 0.5:
                ksize = random.choice([7, 9, 11])
                bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

            # Ghi file ảnh và thông tin nhãn Ground Truth
            file_name = f"{cat}_{i:03d}.jpg"
            save_path = out_base / cat / file_name
            cv2.imwrite(str(save_path), bg)

            all_gt[f"{cat}/{file_name}"] = {
                "corners": final_corners.tolist(),
                "category": cat,
                "is_occluded": (cat == "4_occlusion")
            }

    # Xuất toàn bộ thông tin Ground Truth ra file gt_annotations.json
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
