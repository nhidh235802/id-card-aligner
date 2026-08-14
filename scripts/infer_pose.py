"""
scripts/infer_pose.py
─────────────────────────────────────────────────────────────────
Script inference nhanh cho YOLO-Pose Card Detector.

Dùng để test model trên 1 ảnh hoặc 1 thư mục, xuất ảnh kết quả
với format đúng chuẩn pipeline: class_name + conf + bbox + 4 keypoints.

Cách dùng:
    # Test 1 ảnh
    python scripts/infer_pose.py --img path/to/image.jpg --weights runs/pose/.../best.pt

    # Test thư mục
    python scripts/infer_pose.py --folder data/real_test --weights runs/pose/.../best.pt --save

    # Dùng pretrained YOLO không fine-tune (để thử nhanh)
    python scripts/infer_pose.py --img path/to/image.jpg --weights yolo11n-pose.pt
"""

import sys
import argparse
import cv2
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.pose_detector import PoseDetector
from src.aligner.perspective_aligner import PerspectiveAligner
from src.utils.vis_utils import draw_detection_result

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_inference(img_path: Path, detector: PoseDetector,
                  aligner: PerspectiveAligner, save_dir: Path,
                  show: bool, save: bool, align: bool) -> None:
    """Chạy inference trên 1 ảnh và hiển thị / lưu kết quả."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  ⚠️  Không đọc được ảnh: {img_path}")
        return

    # ── Bước 1: Detect 4 góc thẻ ─────────────────────────────────────────────
    result = detector.detect(image)

    if result.confidence < 0.01:
        print(f"  ❌ Không phát hiện thẻ trong: {img_path.name}")
        return

    print(f"  ✅ {img_path.name:<40}  "
          f"class={result.class_name:<14}  "
          f"conf={result.confidence:.3f}  "
          f"{'[occluded]' if result.is_occluded else ''}")

    # ── Bước 2: Vẽ kết quả detection (class + conf + bbox + keypoints) ───────
    vis = draw_detection_result(image, result, show_corner_labels=True)

    # ── Bước 3: Nếu yêu cầu align — warp perspective → ảnh phẳng ────────────
    if align:
        aligned = aligner.align(image, result.corners)
        # Ghép ảnh gốc có annotation + ảnh aligned cạnh nhau
        import numpy as np
        h_vis = vis.shape[0]
        scale = h_vis / aligned.shape[0]
        w_ali = int(aligned.shape[1] * scale)
        aligned_rs = cv2.resize(aligned, (w_ali, h_vis))
        divider = np.full((h_vis, 6, 3), 255, dtype=np.uint8)
        vis = np.hstack([vis, divider, aligned_rs])

    # ── Bước 4: Lưu / hiển thị ───────────────────────────────────────────────
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"infer_{img_path.name}"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"           → Saved: {out_path}")

    if show:
        win_name = f"[YOLO-Pose] {img_path.name}"
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(0)
        cv2.destroyWindow(win_name)
        if key == 27:       # ESC → thoát sớm
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Inference YOLO-Pose Card Detector — xuất ảnh với class + conf + bbox + keypoints"
    )

    # Input
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--img",    type=str, help="Đường dẫn 1 ảnh đơn lẻ")
    input_grp.add_argument("--folder", type=str, help="Thư mục chứa nhiều ảnh")

    # Model
    parser.add_argument("--weights",  type=str, default="yolo11n-pose.pt",
                        help="Đường dẫn file weights .pt (mặc định: yolo11n-pose.pt)")
    parser.add_argument("--conf",     type=float, default=0.25,
                        help="Ngưỡng confidence (mặc định: 0.25)")
    parser.add_argument("--imgsz",    type=int, default=640,
                        help="Kích thước ảnh đầu vào (mặc định: 640)")

    # Output
    parser.add_argument("--save",     action="store_true",
                        help="Lưu ảnh kết quả vào outputs/infer/")
    parser.add_argument("--show",     action="store_true",
                        help="Hiển thị cửa sổ xem ảnh (nhấn phím bất kỳ để next, ESC để thoát)")
    parser.add_argument("--align",    action="store_true",
                        help="Warp perspective → ghép ảnh aligned bên phải")
    parser.add_argument("--out_dir",  type=str, default="outputs/infer",
                        help="Thư mục lưu kết quả (mặc định: outputs/infer)")
    parser.add_argument("--num",      type=int, default=100,
                        help="Số ảnh tối đa khi dùng --folder")

    args = parser.parse_args()

    # ── Khởi tạo model ────────────────────────────────────────────────────────
    weights = args.weights
    if not Path(weights).exists():
        print(f"\n❌ Không tìm thấy file weights: {weights}")
        print("   Hãy chỉ định đúng đường dẫn với --weights <path>\n")
        sys.exit(1)

    print(f"\n🚀 YOLO-Pose Inference")
    print(f"   weights : {weights}")
    print(f"   conf    : {args.conf}")
    print(f"   imgsz   : {args.imgsz}\n")

    cfg = {
        "weights":          weights,
        "conf_threshold":   args.conf,
        "iou_threshold":    0.45,
        "imgsz":            args.imgsz,
        "use_subpixel":     False,   # Tắt sub-pixel khi demo để nhanh
        "occlusion_min_conf": 0.3,
    }
    detector = PoseDetector(cfg).load_model()
    aligner  = PerspectiveAligner(target_width=856, target_height=540)
    save_dir = Path(args.out_dir)

    # ── Thu thập danh sách ảnh cần xử lý ─────────────────────────────────────
    if args.img:
        img_list = [Path(args.img)]
    else:
        folder = Path(args.folder)
        img_list = sorted(p for p in folder.rglob("*")
                          if p.suffix.lower() in IMG_EXTS)[:args.num]
        print(f"📁 Folder: {folder}  ({len(img_list)} ảnh)\n")

    # ── Chạy inference từng ảnh ───────────────────────────────────────────────
    for img_path in img_list:
        run_inference(img_path, detector, aligner,
                      save_dir=save_dir,
                      show=args.show,
                      save=args.save or not args.show,
                      align=args.align)

    if args.save or not args.show:
        print(f"\n📂 Kết quả lưu tại: {save_dir.resolve()}\n")


if __name__ == "__main__":
    main()
