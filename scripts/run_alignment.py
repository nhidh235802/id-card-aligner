import sys
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

# Đảm bảo PYTHONPATH trỏ đúng vào thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligner.perspective_aligner import PerspectiveAligner
from src.utils.corner_utils import order_corners

DEFAULT_OBB_WEIGHTS = "runs/obb/runs/train/obb_finetune/weights/best.pt"
DEFAULT_POSE_WEIGHTS = "runs/pose/runs/train/pose_finetune/weights/best.pt"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Bộ nạp các lớp Detector ──────────────────────────────────────────────────

def load_detector(detector_name: str, obb_weights: str, pose_weights: str):
    """Khởi tạo và nạp trọng số tương ứng cho Classical, OBB hoặc Pose Detector."""
    if detector_name == "classical":
        with open("configs/classical_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from src.detector.classical_detector import ClassicalDetector
        return ClassicalDetector(cfg).load_model()

    elif detector_name == "obb":
        with open("configs/obb_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["weights"] = obb_weights
        from src.detector.obb_detector import OBBDetector
        return OBBDetector(cfg).load_model()

    elif detector_name == "pose":
        with open("configs/pose_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["weights"] = pose_weights
        from src.detector.pose_detector import PoseDetector
        return PoseDetector(cfg).load_model()

    else:
        raise ValueError(f"Detector không hợp lệ: '{detector_name}'. Chọn: classical / obb / pose")


# ── Hàm vẽ và trực quan hóa kết quả ──────────────────────────────────────────

def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Vẽ 4 vị trí điểm góc thẻ (TL, TR, BR, BL) và đường khung đa giác bao quanh lên ảnh gốc."""
    vis = image.copy()
    pts = corners.astype(np.int32)
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    # Vẽ polygon viền khung thẻ
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 200, 0), thickness=2)

    # Vẽ hình tròn và chữ đánh dấu từng điểm góc
    for i, (pt, label, color) in enumerate(zip(pts, labels, colors)):
        cv2.circle(vis, tuple(pt), 8, color, -1)
        cv2.putText(vis, label, (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def make_comparison(original_with_corners: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Ghép ảnh gốc (có vẽ góc) và ảnh nắn phẳng cạnh nhau theo cùng chiều cao để so sánh."""
    h_orig, w_orig = original_with_corners.shape[:2]
    h_ali,  w_ali  = aligned.shape[:2]

    # Resize ảnh nắn phẳng về cùng chiều cao với ảnh gốc
    scale = h_orig / h_ali if h_ali > 0 else 1.0
    new_w = int(w_ali * scale)
    aligned_resized = cv2.resize(aligned, (new_w, h_orig), interpolation=cv2.INTER_AREA)

    # Vẽ tiêu đề nhãn lên 2 ảnh
    orig_label = original_with_corners.copy()
    cv2.putText(orig_label, "ORIGINAL (4 goc detected)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    ali_label = aligned_resized.copy()
    cv2.putText(ali_label, "ALIGNED (ISO ID-1)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Đường kẻ dọc ngăn cách giữa 2 ảnh
    divider = np.full((h_orig, 4, 3), 128, dtype=np.uint8)
    return np.hstack([orig_label, divider, ali_label])


# ── Pipeline xử lý chính ───────────────────────────────────────────────────────

def process_image(image_path: Path, detector, aligner: PerspectiveAligner,
                  conf_threshold: float, show: bool, show_comparison: bool,
                  save: bool, save_dir: Path) -> dict:
    """Chạy quy trình nắn phẳng cho 1 ảnh: Đọc ảnh → Phát hiện 4 góc → Perspective Warp → Lưu/Hiển thị."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ⚠️  Không đọc được ảnh: {image_path}")
        return {"success": False, "img_path": str(image_path)}

    # Khởi chạy detector phát hiện 4 góc thẻ
    result = detector.detect(image)

    if result.confidence < conf_threshold:
        print(f"  ❌ [{image_path.name}] Không phát hiện được thẻ (conf={result.confidence:.2f} < {conf_threshold})")
        return {"success": False, "img_path": str(image_path)}

    # Thực hiện perspective align với 4 góc thu được
    corners = result.corners
    aligned = aligner.align(image, corners)

    print(f"  ✅ [{image_path.name}] Detect thành công (conf={result.confidence:.2f}) → aligned {aligned.shape[1]}×{aligned.shape[0]}px")

    # Hiển thị kết quả trực quan trên màn hình nếu bật tham số --show hoặc --show-comparison
    if show or show_comparison:
        vis_original = draw_corners(image, corners)
        if show_comparison:
            display = make_comparison(vis_original, aligned)
            cv2.imshow(f"Alignment — {image_path.name}", display)
        else:
            cv2.imshow(f"Original — {image_path.name}", vis_original)
            cv2.imshow(f"Aligned  — {image_path.name}", aligned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Lưu ảnh nắn phẳng ra đĩa nếu bật tham số --save
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"aligned_{image_path.stem}.jpg"
        cv2.imwrite(str(out_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"     → Đã lưu: {out_path}")

    return {"success": True, "img_path": str(image_path)}


def main():
    """Điểm điều khiển chính cho script run_alignment.py."""
    parser = argparse.ArgumentParser(
        description="ID Card Aligner — Pipeline Detect + Warp",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Đọc đối số nguồn ảnh (1 ảnh hoặc 1 thư mục)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--img",    type=str, help="Đường dẫn tới 1 ảnh đơn lẻ")
    input_group.add_argument("--folder", type=str, help="Đường dẫn thư mục chứa nhiều ảnh")

    # Cấu hình detector và weights
    parser.add_argument("--detector",    default="obb",
                        choices=["classical", "obb", "pose"],
                        help="Phương pháp phát hiện góc (mặc định: obb)")
    parser.add_argument("--obb_weights", default=DEFAULT_OBB_WEIGHTS)
    parser.add_argument("--pose_weights",default=DEFAULT_POSE_WEIGHTS)
    parser.add_argument("--conf",        type=float, default=0.30,
                        help="Ngưỡng confidence tối thiểu để chấp nhận detection")

    # Cấu hình lưu và hiển thị đầu ra
    parser.add_argument("--show",            action="store_true",
                        help="Hiển thị cửa sổ ảnh gốc và ảnh aligned")
    parser.add_argument("--show-comparison", action="store_true",
                        help="Hiển thị side-by-side: gốc (có vẽ góc) + aligned")
    parser.add_argument("--save",            action="store_true",
                        help="Lưu ảnh aligned vào thư mục outputs/aligned/")
    parser.add_argument("--save_dir",        default="outputs/aligned",
                        help="Thư mục lưu ảnh aligned (mặc định: outputs/aligned/)")
    parser.add_argument("--output_size",     default="856x540",
                        help="Kích thước ảnh output, VD: 856x540 (mặc định)")

    args = parser.parse_args()

    # Parse kích thước ảnh thẻ đầu ra
    try:
        out_w, out_h = map(int, args.output_size.split("x"))
    except ValueError:
        print(f"❌ Định dạng output_size không hợp lệ: '{args.output_size}'. Dùng định dạng WxH, VD: 856x540")
        sys.exit(1)

    print(f"\n📌 Detector : {args.detector.upper()}")
    print(f"📐 Output   : {out_w}×{out_h} px (ISO ID-1)")
    print(f"🎯 Conf     : ≥ {args.conf}\n")

    # Load mô hình phát hiện và bộ nắn phẳng
    detector = load_detector(args.detector, args.obb_weights, args.pose_weights)
    aligner  = PerspectiveAligner(target_width=out_w, target_height=out_h)
    save_dir = Path(args.save_dir)

    # Thu thập danh sách ảnh đầu vào
    if args.img:
        image_paths = [Path(args.img)]
    else:
        folder = Path(args.folder)
        image_paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])
        print(f"📁 Tìm thấy {len(image_paths)} ảnh trong '{folder}'\n")

    # Lặp qua từng ảnh và tiến hành pipeline
    results = []
    for img_path in image_paths:
        r = process_image(img_path, detector, aligner, args.conf,
                          args.show, args.show_comparison, args.save, save_dir)
        results.append(r)

    # In kết quả thống kê tổng thể
    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*55}")
    print(f"  Tổng kết: {n_success}/{len(results)} ảnh align thành công")
    if args.save:
        print(f"  Ảnh đã lưu tại: {save_dir.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
