import sys
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligner.perspective_aligner import PerspectiveAligner
from src.utils.vis_utils import draw_detection_result

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Trọng số của phiên bản cũ (OLD)
OLD_WEIGHTS = {
    "obb": "runs/obb/runs/train/obb_finetune/weights/best.pt",
    "pose": "runs/pose/runs/train/pose_finetune/weights/best.pt",
}

# Trọng số của phiên bản mới (NEW)
NEW_WEIGHTS = {
    "obb": "runs/obb/runs_new/obb_finetune/weights/best.pt",
    "pose": "C:/Users/Admin/ket-qua-train-mixed-v1/weights/best.pt",
}


def load_detector(detector_type: str, weights_path: str):
    """Khởi tạo và nạp trọng số cho bộ phát hiện góc thẻ (Classical, OBB hoặc Pose)."""
    if detector_type == "classical":
        with open("configs/classical_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from src.detector.classical_detector import ClassicalDetector
        return ClassicalDetector(cfg).load_model()
    elif detector_type == "obb":
        with open("configs/obb_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["weights"] = weights_path
        from src.detector.obb_detector import OBBDetector
        return OBBDetector(cfg).load_model()
    elif detector_type == "pose":
        with open("configs/pose_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["weights"] = weights_path
        from src.detector.pose_detector import PoseDetector
        return PoseDetector(cfg).load_model()
    else:
        raise ValueError(f"Detector không hợp lệ: {detector_type}")


def main():
    """Hàm chính điều khiển quá trình visual debug alignment."""
    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Visual Debug Alignment: Vẽ 4 góc detected + ảnh aligned cạnh nhau")
    parser.add_argument("--folder", default="data_new/real_test", help="Thư mục ảnh cần debug")
    parser.add_argument("--ver", choices=["new", "old"], default="new", help="Chọn phiên bản model: new hoặc old")
    parser.add_argument("--detector", choices=["obb", "pose", "classical"], default="obb", help="Loại detector")
    parser.add_argument("--weights", default=None, help="Ghi đè đường dẫn weights")
    parser.add_argument("--num", type=int, default=30, help="Số lượng ảnh tối đa")
    parser.add_argument("--show_fail", action="store_true",
                        help="Lưu cả ảnh bị SKIP (box OK nhưng kpt conf thấp) — dùng cho báo cáo")
    parser.add_argument("--out_dir", default=None, help="Thư mục xuất ảnh (mặc định: outputs/debug_align_{ver})")
    args = parser.parse_args()

    version = args.ver.lower()

    # Xác định đường dẫn file trọng số weights
    if args.weights:
        weights_path = args.weights
    elif version == "new":
        weights_path = NEW_WEIGHTS.get(args.detector)
        if args.folder == "data_new/real_test" and not Path(args.folder).exists():
            args.folder = "data/real_test"
    else:
        weights_path = OLD_WEIGHTS.get(args.detector)
        if args.folder == "data_new/real_test":
            args.folder = "data/real_test"

    # Kiểm tra sự tồn tại của file weights
    if args.detector in ["obb", "pose"] and weights_path and not Path(weights_path).exists():
        print(f"\n❌ Không tìm thấy weights ({weights_path})!")
        print(f"👉 Hãy train model trước hoặc chỉ định --ver old / --weights\n")
        return

    folder = Path(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/debug_align_{version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách đường dẫn các ảnh cần chạy debug
    images = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])[:args.num]

    print(f"\n🔍 Visual Debug Alignment | Detector: {args.detector.upper()} ({version.upper()})")
    print(f"📁 Folder   : {folder}")
    if weights_path:
        print(f"🎯 Weights  : {weights_path}")
    print(f"📂 Save dir : {out_dir.resolve()}\n")

    # Khởi tạo mô hình detector và đối tượng nắn thẳng aligner
    detector = load_detector(args.detector, weights_path)
    aligner  = PerspectiveAligner(target_width=856, target_height=540)

    # Duyệt và xử lý từng ảnh
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Phát hiện 4 góc thẻ
        result = detector.detect(image)
        box_conf  = result.confidence
        no_box    = box_conf < 0.10 or np.all(result.corners == 0)

        # Chế độ thông thường: bỏ qua nếu conf thấp hoặc bị che
        if not args.show_fail:
            if box_conf < 0.25 or np.all(result.corners == 0):
                print(f"  ❌ SKIP {img_path.name}: conf={result.confidence:.2f}, occluded={result.is_occluded}")
                continue

            corners = result.corners
            vis = draw_detection_result(image, result, show_corner_labels=True)
            aligned = aligner.align(image, corners)

            h_orig = vis.shape[0]
            scale  = h_orig / aligned.shape[0]
            aligned_resized = cv2.resize(aligned, (int(aligned.shape[1] * scale), h_orig))
            divider  = np.full((h_orig, 6, 3), 255, dtype=np.uint8)
            combined = np.hstack([vis, divider, aligned_resized])

            save_path = out_dir / f"debug_{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  ✅ {img_path.name}  class={result.class_name}  conf={result.confidence:.2f}")

        else:
            # ── Chế độ --show_fail: luôn vẽ kết quả để minh hoạ cho báo cáo ──
            if no_box:
                print(f"  ⛔ NO-BOX {img_path.name}: box_conf={box_conf:.2f}")
                continue

            vis = draw_detection_result(image, result, show_corner_labels=True)

            # Lấy keypoint conf trực tiếp từ raw result nếu có
            kpt_min_conf = 0.0
            if hasattr(result, '_raw_kpt_conf') and result._raw_kpt_conf is not None:
                kpt_min_conf = float(result._raw_kpt_conf.min())
            elif hasattr(result, 'corner_confidences') and result.corner_confidences is not None and len(result.corner_confidences) > 0:
                kpt_min_conf = float(np.min(result.corner_confidences))

            # Ghi nhãn trạng thái lên góc trên ảnh
            is_bad_kpt = result.is_occluded or np.all(result.corners == 0)
            status_txt = (f"BOX OK ({box_conf:.2f}) | KPT FAIL (kpt_conf_min={kpt_min_conf:.2f})"
                          if is_bad_kpt
                          else f"BOX OK ({box_conf:.2f}) | KPT OK")
            color = (0, 0, 220) if is_bad_kpt else (0, 180, 0)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (30, 30, 30), -1)
            cv2.putText(vis, status_txt, (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Nếu keypoint tệ: chỉ lưu ảnh phát hiện, không warp
            if is_bad_kpt:
                save_path = out_dir / f"fail_{img_path.stem}.jpg"
                cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"  ⚠️  FAIL-KPT {img_path.name}  box={box_conf:.2f}  kpt_min={kpt_min_conf:.2f}")
            else:
                corners = result.corners
                aligned = aligner.align(image, corners)
                h_orig = vis.shape[0]
                scale  = h_orig / aligned.shape[0]
                aligned_resized = cv2.resize(aligned, (int(aligned.shape[1] * scale), h_orig))
                divider  = np.full((h_orig, 6, 3), 255, dtype=np.uint8)
                combined = np.hstack([vis, divider, aligned_resized])
                save_path = out_dir / f"ok_{img_path.stem}.jpg"
                cv2.imwrite(str(save_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"  ✅ OK {img_path.name}  box={box_conf:.2f}")

    print(f"\n🎉 Hoàn thành! Kiểm tra ảnh debug tại: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
